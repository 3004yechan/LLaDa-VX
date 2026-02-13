#!/usr/bin/env python3
"""
Convert a FIMX-style single-turn JSON (with answer/explanation fields)
into a LLaDA-V standard JSON where the assistant turn is fully populated
using the same logic as LazySupervisedDatasetForFIMX.

Example:
    python tools/convert_fimx_to_filled_llada.py \\
        --input train/data/fimx_raw.json \\
        --output train/data/fimx_filled.json \\
        --tokenizer /home/20223206/model/LLaDA-V-HF \\
        --answer-block-size 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from transformers import AutoTokenizer

DEFAULT_RESERVED_TOKEN = "<|reserved_token_1|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to FIMX-style JSON (list or id->sample dict).")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the filled JSON.")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer path or model id.")
    parser.add_argument("--answer-block-size", type=int, default=20, help="Number of reserved tokens to allocate for the answer prefix+text.")
    parser.add_argument(
        "--reserved-token",
        type=str,
        default=DEFAULT_RESERVED_TOKEN,
        help="Reserved slot token used by the model (must exist in the tokenizer).",
    )
    parser.add_argument(
        "--explanation-field",
        type=str,
        default="explanation",
        help="Field name containing explanation text (string or list).",
    )
    parser.add_argument(
        "--answer-field",
        type=str,
        default="answer",
        help="Field name containing the final answer text.",
    )
    parser.add_argument(
        "--strip-answer-fields",
        action="store_true",
        help="Drop answer/explanation fields in the output (only keep filled conversations).",
    )
    return parser.parse_args()


def load_payload(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.values())
    raise ValueError(f"Unsupported JSON structure: {type(payload)}")


def ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        return text + "."
    return text


def pick_explanation(sample: Dict[str, Any], field: str) -> str:
    value = sample.get(field, "")
    if isinstance(value, list):
        for item in value:
            if item and str(item).strip():
                return ensure_sentence(str(item).strip())
        return ""
    if isinstance(value, str):
        return ensure_sentence(value)
    return ""


def pick_answer(sample: Dict[str, Any], field: str) -> str:
    value = sample.get(field, "") or sample.get("answers", "")
    return str(value).strip()


def build_assistant_text(
    tokenizer,
    answer: str,
    explanation: str,
    answer_block_size: int,
    reserved_token: str,
) -> str:
    if answer_block_size <= 0:
        raise ValueError("answer_block_size must be > 0")

    reserved_id = tokenizer.convert_tokens_to_ids(reserved_token)
    if reserved_id is None or reserved_id < 0:
        raise ValueError(f"Tokenizer is missing reserved token {reserved_token}")

    prefix_ids = tokenizer("The answer is ", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids if answer else []

    block_ids = [reserved_id] * answer_block_size
    copy_len = min(len(prefix_ids), answer_block_size)
    block_ids[:copy_len] = prefix_ids[:copy_len]

    remaining = answer_block_size - copy_len
    if remaining > 0 and answer_ids:
        ans_copy_len = min(len(answer_ids), remaining)
        block_ids[copy_len : copy_len + ans_copy_len] = answer_ids[:ans_copy_len]

    explanation_ids: List[int] = []
    if explanation:
        because_ids = tokenizer(" because ", add_special_tokens=False).input_ids
        explanation_body_ids = tokenizer(explanation, add_special_tokens=False).input_ids
        explanation_ids = because_ids + explanation_body_ids

    label_ids = block_ids + explanation_ids
    # Decode and re-tokenize to ensure the reserved block is at least `answer_block_size` long.
    # Detok/retok can merge spaces, shrinking the block. If it happens, pad more reserved tokens.
    def _block_len(ids):
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    while _block_len(block_ids) < answer_block_size:
        block_ids.append(reserved_id)

    label_ids = block_ids + explanation_ids
    return tokenizer.decode(label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def fill_sample(
    sample: Dict[str, Any],
    tokenizer,
    answer_block_size: int,
    reserved_token: str,
    explanation_field: str,
    answer_field: str,
    strip_answer_fields: bool,
) -> Dict[str, Any]:
    if "conversations" not in sample or len(sample["conversations"]) < 2:
        raise ValueError("Sample is missing a 2-turn conversation.")
    if sample["conversations"][0].get("from") != "human":
        raise ValueError("First conversation turn must be from 'human'.")
    if sample["conversations"][1].get("from") != "gpt":
        raise ValueError("Second conversation turn must be from 'gpt'.")

    answer = pick_answer(sample, answer_field)
    if not answer:
        raise ValueError("Answer text is empty.")
    explanation = pick_explanation(sample, explanation_field)

    assistant_text = build_assistant_text(
        tokenizer=tokenizer,
        answer=answer,
        explanation=explanation,
        answer_block_size=answer_block_size,
        reserved_token=reserved_token,
    )

    conversations = list(sample["conversations"])
    conversations[1] = dict(conversations[1], value=assistant_text)

    output = dict(sample)
    output["conversations"] = conversations
    if strip_answer_fields:
        output.pop(explanation_field, None)
        output.pop(answer_field, None)
        output.pop("answers", None)
    return output


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(load_payload(args.input)):
        try:
            records.append(
                fill_sample(
                    sample=sample,
                    tokenizer=tokenizer,
                    answer_block_size=args.answer_block_size,
                    reserved_token=args.reserved_token,
                    explanation_field=args.explanation_field,
                    answer_field=args.answer_field,
                    strip_answer_fields=args.strip_answer_fields,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] skipping sample {idx}: {exc}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} samples to {args.output}")


if __name__ == "__main__":
    main()
