import argparse

from llava.train.train import DataArguments, LazySupervisedDatasetForFIMX
from transformers import AutoTokenizer


def build_dataset(
    data_path: str,
    image_folder: str,
    tokenizer,
    answer_block_size: int,
    fimx_explanation_first: bool,
    fimx_explanation_block_size: int,
):
    data_args = DataArguments(
        data_path=data_path,
        image_folder=image_folder,
        image_aspect_ratio="anyres_max_4",
        fimx_explanation_first=fimx_explanation_first,
        fimx_explanation_block_size=fimx_explanation_block_size,
    )
    return LazySupervisedDatasetForFIMX(
        data_path=data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        answer_block_size=answer_block_size,
    )


def inspect_one(ds, tokenizer, title: str):
    raw = ds.list_data_dict[0]
    label_ids = ds._build_label_ids(raw)
    label_text = tokenizer.decode(
        label_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    print(f"\n=== {title} ===")
    print("id:", raw.get("id", 0))
    print("human turn:", raw["conversations"][0]["value"])
    print("answer field:", raw.get("answer") or raw.get("answers"))
    print("explanation field:", raw.get("explanation"))
    print("constructed label_text:", label_text)
    print("constructed label_ids:", label_ids[:60])


def main():
    parser = argparse.ArgumentParser(description="Inspect FIMX label construction.")
    parser.add_argument("--model-dir", default="/workspace/model/LLaDA-V-HF")
    parser.add_argument("--data-path", default="/workspace/ACT-X/actX_train_filled_llada.json")
    parser.add_argument("--image-folder", default="/workspace/ACT-X/images")
    parser.add_argument("--answer-block-size", type=int, default=20)
    parser.add_argument("--explanation-block-size", type=int, default=70)
    parser.add_argument(
        "--fimx-explanation-first",
        action="store_true",
        help="Use explanation-first format: 'Because ..., the answer is ...'.",
    )
    parser.add_argument(
        "--compare-both",
        action="store_true",
        help="Print both answer-first and explanation-first for comparison.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)

    if args.compare_both:
        ds_answer_first = build_dataset(
            args.data_path,
            args.image_folder,
            tokenizer,
            args.answer_block_size,
            fimx_explanation_first=False,
            fimx_explanation_block_size=args.explanation_block_size,
        )
        inspect_one(ds_answer_first, tokenizer, "answer-first")

        ds_explanation_first = build_dataset(
            args.data_path,
            args.image_folder,
            tokenizer,
            args.answer_block_size,
            fimx_explanation_first=True,
            fimx_explanation_block_size=args.explanation_block_size,
        )
        inspect_one(ds_explanation_first, tokenizer, "explanation-first")
        return

    ds = build_dataset(
        args.data_path,
        args.image_folder,
        tokenizer,
        args.answer_block_size,
        fimx_explanation_first=args.fimx_explanation_first,
        fimx_explanation_block_size=args.explanation_block_size,
    )
    mode = "explanation-first" if args.fimx_explanation_first else "answer-first"
    inspect_one(ds, tokenizer, mode)


if __name__ == "__main__":
    main()
