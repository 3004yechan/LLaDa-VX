#!/usr/bin/env python3
"""
Convert ACT-X train/test JSON files into LazySupervisedDataset-compatible JSON.

Output format (per sample):
{
  "id": "...",
  "image": "images/xxx.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\\n...question..."},
    {"from": "gpt", "value": "Because ... , the answer is ... ."}
  ]
}

ACT-X has one answer string and typically three explanations per image.
This converter emits one sample per explanation, so each image can produce 3 samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_QUESTION = "What activity is the person (or people) performing in this image? Before you answer, please explain the reason first using the format below: 'Because..., the answer is...'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("/workspace/ACT-X/actX_train.json"),
        help="Path to ACT-X train JSON.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=Path("/workspace/ACT-X/actX_test.json"),
        help="Path to ACT-X test JSON.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("/workspace/ACT-X/actX_train_lazy.json"),
        help="Output JSON path for converted train split.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("/workspace/ACT-X/actX_test_lazy.json"),
        help="Output JSON path for converted test split.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question appended after the <image> token in the human turn.",
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default="images",
        help="Prefix prepended to image_name (e.g., 'images').",
    )
    return parser.parse_args()


def load_actx(path: Path) -> Iterable[Tuple[str, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.items()
    if isinstance(payload, list):
        return ((str(i), sample) for i, sample in enumerate(payload))
    raise ValueError(f"Unsupported ACT-X payload type: {type(payload)}")


def clean_explanation(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    # Remove terminal punctuation to keep a single, stable sentence template.
    while text and text[-1] in ".!?":
        text = text[:-1].rstrip()
    return text


def build_gpt_text(answer: str, explanation: str) -> str:
    answer = (answer or "").strip()
    explanation = clean_explanation(explanation)
    if explanation:
        return f"Because {explanation}, the answer is {answer}."
    return f"The answer is {answer}."


def convert_split(input_path: Path, output_path: Path, question: str, image_prefix: str) -> None:
    records: List[Dict] = []
    num_images = 0
    num_explanations = 0

    for key, sample in load_actx(input_path):
        answer = (sample.get("answers") or "").strip()
        if not answer:
            # Skip malformed records.
            continue

        explanations = sample.get("explanation")
        if isinstance(explanations, list):
            cleaned = [x.strip() for x in explanations if x and x.strip()]
        elif isinstance(explanations, str):
            cleaned = [explanations.strip()] if explanations.strip() else []
        else:
            cleaned = []
        if not cleaned:
            cleaned = [""]

        image_name = sample.get("image_name") or f"{key}.jpg"
        image_id = sample.get("image_id") or key
        image_rel = f"{image_prefix}/{image_name}" if image_prefix else image_name

        num_images += 1
        num_explanations += len(cleaned)
        for idx, exp in enumerate(cleaned):
            entry_id = image_id if len(cleaned) == 1 else f"{image_id}_{idx}"
            records.append(
                {
                    "id": entry_id,
                    "image": image_rel.replace("\\", "/"),
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{question.strip()}"},
                        {"from": "gpt", "value": build_gpt_text(answer, exp)},
                    ],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] {input_path} -> {output_path} | images={num_images}, "
        f"samples={len(records)}, avg_per_image={(len(records) / max(num_images, 1)):.2f}"
    )


def main() -> None:
    args = parse_args()
    convert_split(args.train_input, args.train_output, args.question, args.image_prefix)
    convert_split(args.test_input, args.test_output, args.question, args.image_prefix)


if __name__ == "__main__":
    main()

