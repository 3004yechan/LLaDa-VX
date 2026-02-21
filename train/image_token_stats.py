#!/usr/bin/env python3
"""
Compute image-token length stats for LLaVA/LLaDA anyres settings.

This script mirrors the token-length logic in:
  - llava/model/llava_arch.py (anyres_max + spatial_unpad path)
  - llava/model/multimodal_encoder/siglip_encoder.py (27x27 per 384 crop)
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image
import torch

from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.llava_arch import unpad_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True, help="Dataset JSON path")
    p.add_argument("--image-root", type=str, required=True, help="Root folder for image files")
    p.add_argument("--image-grid-pinpoints", type=str, default="(1x1),...,(6x6)")
    p.add_argument("--image-aspect-ratio", type=str, default="anyres_max_4")
    p.add_argument("--mm-patch-merge-type", type=str, default="spatial_unpad")
    p.add_argument("--vision-image-size", type=int, default=384, help="Vision tower input size")
    p.add_argument("--vision-patch-size", type=int, default=14, help="Vision patch size")
    p.add_argument(
        "--use-real-siglip-config",
        action="store_true",
        help="Load vision config from HuggingFace model/config instead of manual size/patch args",
    )
    p.add_argument(
        "--vision-model-name-or-path",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Model name/path used when --use-real-siglip-config is enabled",
    )
    p.add_argument("--dedup-images", action="store_true", help="Count each image path once")
    p.add_argument("--limit", type=int, default=0, help="If > 0, scan only first N samples")
    return p.parse_args()


def load_image_paths(data_path: Path) -> List[str]:
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_paths: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "image" in item:
                image_paths.append(str(item["image"]))
    elif isinstance(data, dict):
        for _, item in data.items():
            if not isinstance(item, dict):
                continue
            if "image_name" in item:
                image_paths.append(str(item["image_name"]))
            elif "image" in item:
                image_paths.append(str(item["image"]))
    else:
        raise ValueError(f"Unsupported dataset JSON type: {type(data)}")

    if not image_paths:
        raise ValueError("No image paths found in dataset JSON")
    return image_paths


def resolve_image_path(root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return root / p


def compute_image_tokens(
    original_size_wh: Tuple[int, int],
    image_grid_pinpoints: str,
    image_aspect_ratio: str,
    mm_patch_merge_type: str,
    vision_image_size: int,
    vision_patch_size: int,
) -> int:
    # SigLIP2-384 patch tokens per crop = (384/14)^2 = 729
    num_patches_per_side = vision_image_size // vision_patch_size
    base_tokens = num_patches_per_side * num_patches_per_side

    # Non-anyres path (kept for completeness)
    if "anyres" not in image_aspect_ratio:
        if "unpad" in mm_patch_merge_type:
            return base_tokens + 1
        return base_tokens

    num_patch_w, num_patch_h = get_anyres_image_grid_shape(
        original_size_wh, image_grid_pinpoints, vision_image_size
    )

    # This is the CxHxW map right before unpad in llava_arch.
    h = num_patch_h * num_patches_per_side
    w = num_patch_w * num_patches_per_side

    # Reuse project unpad behavior exactly.
    dummy = torch.zeros((1, h, w), dtype=torch.float32)
    unpadded = unpad_image(dummy, original_size_wh)
    _, h2, w2 = unpadded.shape

    matched = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
    if "unpad" in mm_patch_merge_type and matched:
        max_num_patches = int(matched.group(1))
        unit = num_patches_per_side
        times = math.sqrt((h2 * w2) / (max_num_patches * unit * unit))
        if times > 1.1:
            h2 = int(h2 // times)
            w2 = int(w2 // times)
        # newline token adds one column
        dynamic_tokens = h2 * (w2 + 1)
    elif "unpad" in mm_patch_merge_type:
        dynamic_tokens = h2 * (w2 + 1)
    else:
        dynamic_tokens = h2 * w2

    # "nobase" would skip these, but default spatial_unpad keeps base.
    if "nobase" in mm_patch_merge_type:
        return dynamic_tokens
    return base_tokens + dynamic_tokens


def percentile(sorted_vals: Sequence[int], p: float) -> float:
    if not sorted_vals:
        raise ValueError("empty sequence")
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    n = len(sorted_vals)
    rank = (p / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    w = rank - lo
    return (1.0 - w) * sorted_vals[lo] + w * sorted_vals[hi]


def summarize(arr: Sequence[int]) -> None:
    s = sorted(arr)
    n = len(s)
    mean = sum(s) / n
    ps = [0, 1, 5, 50, 95, 99, 100]
    print("Image-token stats:")
    for p in ps:
        print(f"  p{p:>3}: {percentile(s, p):.2f}")
    print(f"  mean: {mean:.2f}")
    print(f"  min : {s[0]}")
    print(f"  max : {s[-1]}")


def resolve_vision_shape_from_model(model_name_or_path: str) -> Tuple[int, int]:
    try:
        from transformers import AutoConfig
    except Exception as e:
        raise RuntimeError(
            "transformers is required for --use-real-siglip-config. "
            "Install it in this environment first."
        ) from e

    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    image_size = getattr(cfg, "image_size", None)
    patch_size = getattr(cfg, "patch_size", None)

    if image_size is None and hasattr(cfg, "vision_config"):
        image_size = getattr(cfg.vision_config, "image_size", None)
        patch_size = getattr(cfg.vision_config, "patch_size", None)

    if image_size is None or patch_size is None:
        raise ValueError(
            f"Could not read image_size/patch_size from config: {model_name_or_path}"
        )
    return int(image_size), int(patch_size)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    image_root = Path(args.image_root)

    vision_image_size = args.vision_image_size
    vision_patch_size = args.vision_patch_size
    if args.use_real_siglip_config:
        vision_image_size, vision_patch_size = resolve_vision_shape_from_model(
            args.vision_model_name_or_path
        )

    print("Vision tokenization setup:")
    print(f"  image_size: {vision_image_size}")
    print(f"  patch_size: {vision_patch_size}")
    if args.use_real_siglip_config:
        print(f"  source: {args.vision_model_name_or_path}")
    else:
        print("  source: manual args")

    rel_paths = load_image_paths(data_path)
    if args.dedup_images:
        rel_paths = sorted(set(rel_paths))
    if args.limit and args.limit > 0:
        rel_paths = rel_paths[: args.limit]

    token_counts: List[int] = []
    missing = 0
    for rp in rel_paths:
        img_path = resolve_image_path(image_root, rp)
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            t = compute_image_tokens(
                original_size_wh=(w, h),
                image_grid_pinpoints=args.image_grid_pinpoints,
                image_aspect_ratio=args.image_aspect_ratio,
                mm_patch_merge_type=args.mm_patch_merge_type,
                vision_image_size=vision_image_size,
                vision_patch_size=vision_patch_size,
            )
            token_counts.append(int(t))
        except FileNotFoundError:
            missing += 1
        except Exception as e:
            print(f"[warn] failed for {img_path}: {e}")

    print(f"Samples scanned: {len(rel_paths)}")
    print(f"Valid images  : {len(token_counts)}")
    print(f"Missing files : {missing}")
    if not token_counts:
        raise RuntimeError("No valid token counts computed")
    summarize(token_counts)


if __name__ == "__main__":
    main()
