import argparse
import time

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare generation behavior with and without draft tokens using the same checkpoint."
    )
    parser.add_argument("--pretrained", type=str, default="./exp/llada_v_lora")
    parser.add_argument("--model-base", type=str, default="/workspace/model/LLaDA-V-HF")
    parser.add_argument("--model-name", type=str, default="llava_llada_lora")
    parser.add_argument("--image", type=str, default="tennis.jpg")
    parser.add_argument(
        "--question",
        type=str,
        default="What activity is the person (or people) performing in this image?",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--explanation-mask", type=int, default=20)
    parser.add_argument("--answer-mask", type=int, default=7)
    parser.add_argument("--reserved-fill", type=int, default=10)
    return parser.parse_args()


def clean_text(text: str) -> str:
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>", 1)[-1]
    if "<|eot_id|>" in text:
        text = text.split("<|eot_id|>", 1)[0]
    return text.strip()


def run_once(
    model,
    tokenizer,
    input_ids,
    image_tensor,
    image_sizes,
    gen_kwargs,
    draft_tokens,
):
    start = time.time()
    kwargs = dict(gen_kwargs)
    if draft_tokens is not None:
        kwargs["draft_tokens"] = draft_tokens
    out = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        **kwargs,
    )
    elapsed = time.time() - start
    text = tokenizer.decode(out[0], skip_special_tokens=False)
    return out, text, elapsed


def main():
    args = parse_args()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        args.model_base,
        args.model_name,
        attn_implementation="sdpa",
        device_map=args.device_map,
    )
    model.eval()

    if hasattr(model.config, "enable_complementary_masking"):
        model.config.enable_complementary_masking = False
    if hasattr(model.config, "enable_semi_complementary_masking"):
        model.config.enable_semi_complementary_masking = False

    image = Image.open(args.image).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_img.to(dtype=torch.float16, device=args.device) for _img in image_tensor]
    image_sizes = [image.size]

    # Match check_eos_bias.py prompt path exactly.
    prompt = f"{DEFAULT_IMAGE_TOKEN}\n{args.question}"

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)

    draft_answer = (
        "The answer is "
        + ("<|mdm_mask|>" * args.answer_mask)
        + ("<|reserved_token_1|>" * args.reserved_fill)
        + " because "
        + ("<|mdm_mask|>" * args.explanation_mask)
    )
    draft_tokens = tokenizer(draft_answer, return_tensors="pt").input_ids.to(input_ids.device)

    # Match check_eos_bias.py generate kwargs.
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "eos_token_id": tokenizer.eos_token_id,
    }
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(gen_kwargs["eos_token_id"], list):
            gen_kwargs["eos_token_id"].append(eot_id)
        else:
            gen_kwargs["eos_token_id"] = [gen_kwargs["eos_token_id"], eot_id]
    except Exception:
        pass

    print("=== CONFIG ===")
    print(f"pretrained={args.pretrained}")
    print(f"prompt={prompt!r}")
    print(f"gen_kwargs={gen_kwargs}")
    print(f"draft_answer={draft_answer}")

    print("\n=== RUN A: NO DRAFT ===")
    out_a, text_a, t_a = run_once(
        model,
        tokenizer,
        input_ids,
        image_tensor,
        image_sizes,
        gen_kwargs,
        None,
    )
    print(f"time={t_a:.4f}s")
    print("raw tokens:", out_a)
    print("raw text:", repr(text_a))
    print("clean text:", repr(clean_text(text_a)))

    print("\n=== RUN B: WITH DRAFT ===")
    out_b, text_b, t_b = run_once(
        model,
        tokenizer,
        input_ids,
        image_tensor,
        image_sizes,
        gen_kwargs,
        draft_tokens,
    )
    print(f"time={t_b:.4f}s")
    print("raw tokens:", out_b)
    print("raw text:", repr(text_b))
    print("clean text:", repr(clean_text(text_b)))


if __name__ == "__main__":
    main()
