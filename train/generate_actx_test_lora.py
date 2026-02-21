# python generate_actx_test_lora.py \
#     --batch-size 2 \
#     --use-draft-tokens \
#     --enable-reserved-collapse \
#     --output-json /home/20223206/ACT-X/LLaDA-VX_actX_p_full_unfiltered_bs2.json

import argparse
import copy
import json
import time
import warnings
from pathlib import Path

import torch
from PIL import Image

from dataclasses import asdict
from llava.cache import dLLMCache, dLLMCacheConfig
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.hooks import register_cache_LLaDA_V
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


warnings.filterwarnings("ignore")

QUESTION = (
    DEFAULT_IMAGE_TOKEN
    + "\nWhat activity is the person (or people) performing in this image? "
    "Before you answer, please explain the reason first using the format below: "
    "'Because..., the answer is...'. You can use <|reserved_token_1|> to trigger "
    "an early stop for a brief explanation."
)
DEFAULT_RESERVED_TOKEN_1_ID = 126085
DEFAULT_MDM_MASK_ID = 126336


def format_seconds(total_seconds):
    total_seconds = max(0, int(total_seconds))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run LoRA inference on ACT-X test split.")
    parser.add_argument(
        "--pretrained",
        default="/home/20223206/exp/llada_v_qlora_actx_single",
        help="LoRA checkpoint path.",
    )
    parser.add_argument(
        "--model-base",
        default="/home/20223206/model/LLaDA-V-HF",
        help="Base model path.",
    )
    parser.add_argument(
        "--model-name",
        default="llava_llada_lora",
        help="Model name passed to load_pretrained_model.",
    )
    parser.add_argument(
        "--test-json",
        default="/home/20223206/ACT-X/actX_test.json",
        help="ACT-X test annotation JSON path.",
    )
    parser.add_argument(
        "--image-dir",
        default="/home/20223206/ACT-X/images",
        help="ACT-X image directory path.",
    )
    parser.add_argument(
        "--output-json",
        default="/home/20223206/ACT-X/actX_p_full_unfiltered.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device.",
    )
    parser.add_argument(
        "--no-load-4bit",
        action="store_true",
        help="Disable 4-bit loading.",
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--prefix-refresh-interval", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument(
        "--enable-reserved-collapse",
        action="store_true",
        help="Collapse mdm_mask runs after reserved_token_1 in final generated sequence.",
    )
    parser.add_argument(
        "--reserved-token-id",
        type=int,
        default=DEFAULT_RESERVED_TOKEN_1_ID,
        help="Token ID for <|reserved_token_1|>.",
    )
    parser.add_argument(
        "--mdm-mask-id",
        type=int,
        default=DEFAULT_MDM_MASK_ID,
        help="Token ID for <|mdm_mask|>.",
    )
    parser.add_argument(
        "--use-draft-tokens",
        action="store_true",
        help="Use draft tokens like generate_demo_lora.py.",
    )
    parser.add_argument("--explanation-max-token", type=int, default=70)
    parser.add_argument("--answer-max-token", type=int, default=30)
    parser.add_argument(
        "--use-fast-dllm",
        action="store_true",
        help="Enable Fast-dLLM hook for speed-up.",
    )
    parser.add_argument(
        "--use-dllm-cache",
        action="store_true",
        help="Enable dLLM-cache hook for speed-up.",
    )
    parser.add_argument("--prompt-interval-steps", type=int, default=25)
    parser.add_argument("--gen-interval-steps", type=int, default=7)
    parser.add_argument("--transfer-ratio", type=float, default=0.25)
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save intermediate results every N samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples per generate() call. Default keeps original behavior.",
    )
    return parser.parse_args()


def load_test_entries(test_json_path):
    with open(test_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return list(payload.items())
    raise ValueError(f"Unsupported ACT-X test payload type: {type(payload)}")


def main():
    args = parse_args()
    if args.use_fast_dllm and args.use_dllm_cache:
        raise ValueError("Choose only one of --use-fast-dllm or --use-dllm-cache.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_size > 1 and (args.use_fast_dllm or args.use_dllm_cache):
        raise ValueError(
            "--batch-size > 1 currently supports only the base generate path. "
            "Disable --use-fast-dllm and --use-dllm-cache for batched inference."
        )

    test_json = Path(args.test_json)
    image_dir = Path(args.image_dir)
    output_json = Path(args.output_json)

    if not test_json.exists():
        raise FileNotFoundError(f"Test JSON not found: {test_json}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        args.model_base,
        args.model_name,
        attn_implementation="sdpa",
        device_map=args.device,
        load_4bit=not args.no_load_4bit,
    )
    model.eval()
    hook_model = model.get_base_model() if hasattr(model, "get_base_model") else model

    if args.use_fast_dllm:
        register_fast_dllm_hook(hook_model)
        print("Fast-dLLM hook enabled")
    elif args.use_dllm_cache:
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=args.prompt_interval_steps,
                    gen_interval_steps=args.gen_interval_steps,
                    transfer_ratio=args.transfer_ratio,
                )
            )
        )
        register_cache_LLaDA_V(hook_model, "model.layers")
        print("dLLM-cache hook enabled")

    conv_template = "llava_llada"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], QUESTION)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(args.device)

    draft_tokens = None
    if args.use_draft_tokens:
        draft_answer = (
            f'Because{"<|mdm_mask|>" * args.explanation_max_token}, '
            f'the answer is{"<|mdm_mask|>" * args.answer_max_token}<|eot_id|>'
        )
        draft_tokens = tokenizer(draft_answer, return_tensors="pt").to(input_ids.device).input_ids

    entries = load_test_entries(test_json)
    total = len(entries)
    results = []
    start_time = time.time()

    processed = 0
    for start in range(0, total, args.batch_size):
        batch_entries = entries[start : start + args.batch_size]
        valid_items = []
        batch_results = [None] * len(batch_entries)

        for local_idx, (image_id, meta) in enumerate(batch_entries):
            image_name = meta.get("image_name", f"{image_id}.jpg")
            image_path = image_dir / image_name
            if not image_path.exists():
                print(f"[WARN] Missing image: {image_path}")
                batch_results[local_idx] = {"image_id": int(image_id), "caption": "", "cont": []}
                continue

            image = Image.open(image_path).convert("RGB")
            valid_items.append((local_idx, image_id, image))

        if valid_items:
            images = [item[2] for item in valid_items]
            image_tensor = process_images(images, image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=args.device) for _image in image_tensor]
            image_sizes = [image.size for image in images]

            batch_n = len(valid_items)
            batch_input_ids = input_ids.repeat(batch_n, 1)

            with torch.no_grad():
                generate_kwargs = {}
                if draft_tokens is not None:
                    generate_kwargs["draft_tokens"] = draft_tokens.repeat(batch_n, 1)
                if args.enable_reserved_collapse:
                    generate_kwargs["enable_reserved_collapse"] = True
                    generate_kwargs["reserved_token_id"] = args.reserved_token_id
                    generate_kwargs["mdm_mask_id"] = args.mdm_mask_id
                cont = model.generate(
                    batch_input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    modalities=["image"] * batch_n,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    tokenizer=tokenizer,
                    stopping_criteria=["<|eot_id|>"],
                    prefix_refresh_interval=args.prefix_refresh_interval,
                    threshold=args.threshold,
                    **generate_kwargs,
                )

            for row_idx, (local_idx, image_id, _) in enumerate(valid_items):
                cont_raw = cont[row_idx].tolist()
                cont_filtered = [token_id for token_id in cont_raw if token_id != args.reserved_token_id]
                caption = tokenizer.decode(cont_filtered, skip_special_tokens=False)
                caption = caption.replace("<|eot_id|>", "").strip()
                batch_results[local_idx] = {"image_id": int(image_id), "caption": caption, "cont": cont_filtered}

        for item in batch_results:
            if item is None:
                continue
            results.append(item)
            processed += 1

        elapsed = time.time() - start_time
        avg = elapsed / processed if processed > 0 else 0
        eta = avg * (total - processed)
        progress = (processed / total) * 100 if total > 0 else 100
        print(
            f"[{processed}/{total}] {progress:.2f}% | elapsed {format_seconds(elapsed)} | eta {format_seconds(eta)}"
        )

        if processed % args.save_every == 0:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[{processed}/{total}] saved to {output_json}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"Done. Saved {len(results)} results to {output_json}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
