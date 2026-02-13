import argparse
import sys

import torch

from llava.model.builder import load_pretrained_model


def count_meta_params(module: torch.nn.Module) -> int:
    return sum(p.is_meta for p in module.parameters())


def main():
    parser = argparse.ArgumentParser(description="Check if any parameters are still meta tensors after loading.")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to LoRA/finetune checkpoint (e.g., ./exp/llada_v_qlora_single)")
    parser.add_argument("--model-base", type=str, required=True, help="Base model path (e.g., /home/20223206/model/LLaDA-V-HF)")
    parser.add_argument("--model-name", type=str, default="llava_llada_lora", help="Model name string used in load_pretrained_model")
    parser.add_argument("--load-4bit", action="store_true", help="Try loading base in 4bit; leave off to load in fp16")
    parser.add_argument("--device-map", type=str, default="auto", help="device_map passed to loader")
    args = parser.parse_args()

    tokenizer, model, _, _ = load_pretrained_model(
        args.pretrained,
        args.model_base,
        args.model_name,
        load_4bit=args.load_4bit,
        device_map=args.device_map,
        attn_implementation="sdpa",
    )

    meta_total = count_meta_params(model)
    vt = getattr(model, "get_vision_tower", lambda: None)()
    meta_vision = count_meta_params(vt.vision_tower) if vt is not None else 0

    print(f"Meta parameters (entire model): {meta_total}")
    print(f"Meta parameters (vision tower): {meta_vision}")

    if meta_total == 0 and meta_vision == 0:
        print("No meta tensors remain; checkpoint loaded correctly.")
    else:
        print("Warning: meta tensors remain. Consider reloading without meta initialization or reloading the vision tower explicitly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
