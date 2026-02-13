import argparse
from typing import Optional

import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def get_base_lm(model):
    """Return the underlying language model to avoid multimodal wrappers."""
    if hasattr(model, "get_model"):
        return model.get_model()
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "base_model"):
        return model.base_model
    return model


def main():
    parser = argparse.ArgumentParser(description="Inspect last-token logits and EOS bias.")
    parser.add_argument("--pretrained", type=str, required=True, help="LoRA/finetune checkpoint path")
    parser.add_argument("--model-base", type=str, required=True, help="Base model path (HF format)")
    parser.add_argument("--model-name", type=str, default="llava_llada_lora", help="Model name string for loader")
    parser.add_argument("--prompt", type=str, default="Hello", help="Text prompt to probe")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for multimodal probing")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference")
    parser.add_argument("--device-map", type=str, default="cuda:0", help="device_map passed to loader")
    parser.add_argument("--topk", type=int, default=10, help="Top-K tokens to display")
    parser.add_argument("--load-4bit", action="store_true", help="Enable 4bit loading (default fp16/bf16)")
    args = parser.parse_args()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained,
        args.model_base,
        args.model_name,
        attn_implementation="sdpa",
        device_map=args.device_map,
        load_4bit=args.load_4bit,
    )
    # Disable complementary masking flags for inference probes.
    if hasattr(model.config, "enable_complementary_masking"):
        model.config.enable_complementary_masking = False
    if hasattr(model.config, "enable_semi_complementary_masking"):
        model.config.enable_semi_complementary_masking = False

    # If image is provided, run multimodal path; otherwise text-only LM probe.
    use_image = args.image is not None

    if use_image:
        image = Image.open(args.image).convert("RGB")
        prompt = args.prompt
        if DEFAULT_IMAGE_TOKEN not in prompt:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_img.to(args.device, dtype=torch.float16) for _img in image_tensor]
        image_sizes = [image.size]

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
        labels = torch.full_like(input_ids, -100)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, images=image_tensor, image_sizes=image_sizes, labels=labels, return_dict=True)
            logits = outputs.logits
        last_logits = logits[0, -1]
        vocab_size = last_logits.size(0)

        print(f"Prompt (multimodal): {prompt!r}")
    else:
        base_model = get_base_lm(model)
        base_model.eval()

        inputs = tokenizer(args.prompt, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = base_model(**inputs, return_dict=True)
            if hasattr(outputs, "logits") and outputs.logits is not None:
                logits = outputs.logits
            else:
                hidden = outputs.last_hidden_state
                logits = model.lm_head(hidden)

        last_logits = logits[0, -1]
        vocab_size = last_logits.size(0)
        print(f"Prompt: {args.prompt!r}")

    topk = torch.topk(last_logits, k=min(args.topk, vocab_size))

    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id < vocab_size:
        eos_logit = last_logits[tokenizer.eos_token_id]
        eos_rank = (last_logits > eos_logit).sum().item() + 1
        print(f"EOS token id: {tokenizer.eos_token_id}")
        print(f"EOS logit: {eos_logit:.4f}, rank: {eos_rank}")
    else:
        print(f"EOS token id {tokenizer.eos_token_id} is out of bounds for vocab size {vocab_size}.")

    print("\nTop tokens:")
    for idx, logit in zip(topk.indices.tolist(), topk.values.tolist()):
        token_str = tokenizer.decode([idx], skip_special_tokens=False)
        print(f"  id={idx:6d} logit={logit:9.4f} token={token_str!r}")

    # Also run a short generate to see actual output.
    gen_kwargs = {
        "max_new_tokens": 64,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
    }
    # Add eot token if available to stop on conversation end.
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(gen_kwargs["eos_token_id"], list):
            gen_kwargs["eos_token_id"].append(eot_id)
        else:
            gen_kwargs["eos_token_id"] = [gen_kwargs["eos_token_id"], eot_id]
    except Exception:
        pass

    if use_image:
        gen_out = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            **gen_kwargs,
        )
    else:
        gen_inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
        gen_out = model.generate(
            gen_inputs["input_ids"],
            attention_mask=gen_inputs.get("attention_mask", None),
            **gen_kwargs,
        )

    print("\nGenerated:")
    print(tokenizer.decode(gen_out[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
