#!/usr/bin/env python3
"""
Utility to convert ACT-X style annotations into the single-turn JSON
format that LLaDA-V / LLaVA expects.

Example usage:
    python tools/convert_actx_to_llada.py \
        --input /home/20223206/ACT-X/actX_train.json \
        --output train/data/actx/train_llada.json \
        --image-prefix images \
        --question "What activity is the person (or people) performing in this image?"

Run the script twice (train / test) or point it to any other ACT-X style
annotation file.  The resulting JSON can be consumed directly by
`train/scripts/llada_v_finetune.sh` via `--data_path`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_IMAGE_TOKEN = "<image>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to ACT-X style JSON file (train/test).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the converted LLaDA-V JSON file.",
    )
    parser.add_argument(
        "--image-prefix",
        default="",
        type=str,
        help=(
            "Optional subdirectory (relative path) to prepend to every image_name. "
            "Use this when your images live inside a folder such as 'images/'."
        ),
    )
    parser.add_argument(
        "--question",
        default="What activity is happening in this image?",
        type=str,
        help="Human turn template that follows the <image> token.",
    )
    parser.add_argument(
        "--image-token-separator",
        default="\n",
        type=str,
        help=(
            "Text inserted immediately after the <image> token before the question/explanation. "
            "Use '\\n' (default) to start the prompt on a new line."
        ),
    )
    parser.add_argument(
        "--split-explanations",
        action="store_true",
        help="Emit one training sample per explanation instead of picking a single one.",
    )
    parser.add_argument(
        "--explanation-strategy",
        default="first",
        choices=("first", "random"),
        help=(
            "How to pick an explanation when --split-explanations is disabled. "
            "'first' uses index 0, 'random' samples uniformly (set --seed)."
        ),
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed when using --explanation-strategy=random.",
    )
    parser.add_argument(
        "--empty-assistant-placeholder",
        default="",
        type=str,
        help=(
            "Value inserted into the assistant turn. Leave empty to emit an empty string. "
            "Downstream code can read the 'answer'/'explanation' fields and assemble the final text."
        ),
    )
    return parser.parse_args()


def load_actx(path: Path) -> Iterable[Tuple[str, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return payload.items()
    elif isinstance(payload, list):
        # Already in list form; assign numeric IDs as fallback.
        return ((str(i), sample) for i, sample in enumerate(payload))
    else:
        raise ValueError(f"Unsupported ACT-X payload type: {type(payload)}")


def pick_explanation(explanations: List[str], strategy: str) -> str:
    if not explanations:
        return ""
    cleaned = [exp.strip() for exp in explanations if exp and exp.strip()]
    if not cleaned:
        return ""
    if strategy == "first":
        return cleaned[0]
    if strategy == "random":
        return random.choice(cleaned)
    if strategy == "concat":
        return " ".join(cleaned)
    raise ValueError(f"Unknown explanation strategy: {strategy}")


def ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    terminal = text[-1]
    if terminal not in ".!?":
        text += "."
    return text


def build_sample(
    sample_id: str,
    sample: Dict,
    args: argparse.Namespace,
) -> List[Dict]:
    answer = (sample.get("answers") or "").strip()
    if not answer:
        raise ValueError(f"Sample {sample_id} is missing 'answers'.")

    explanations = sample.get("explanation", []) or [""]
    cleaned_explanations = [exp.strip() for exp in explanations if exp and exp.strip()]
    if not cleaned_explanations:
        cleaned_explanations = [""]

    if not args.split_explanations:
        explanation = pick_explanation(cleaned_explanations, args.explanation_strategy)
        cleaned_explanations = [explanation]

    question_text = args.question.strip()
    if not question_text:
        raise ValueError("Question template cannot be empty.")

    image_name = sample.get("image_name") or f"{sample_id}.jpg"
    image_path = Path(args.image_prefix) / image_name if args.image_prefix else Path(image_name)

    outputs: List[Dict] = []
    base_id = sample.get("image_id", sample_id)

    for idx, explanation in enumerate(cleaned_explanations):
        separator = "\n" if args.image_token_separator == r"\n" else args.image_token_separator
        conversations = [
            {
                "from": "human",
                "value": f"{DEFAULT_IMAGE_TOKEN}{separator}{question_text}",
            },
            {
                "from": "gpt",
                "value": args.empty_assistant_placeholder,
            },
        ]

        entry_id = base_id if idx == 0 else f"{base_id}_{idx}"
        outputs.append(
            {
                "id": entry_id,
                "image": str(image_path).replace("\\", "/"),
                "answer": answer,
                "explanation": explanation,
                "explanation_index": idx,
                "conversations": conversations,
            }
        )

    return outputs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    records: List[Dict] = []
    for key, sample in load_actx(args.input):
        try:
            records.extend(build_sample(key, sample, args))
        except ValueError as exc:
            print(f"[WARN] skipping sample {key}: {exc}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} samples to {args.output}")


if __name__ == "__main__":
    main()
