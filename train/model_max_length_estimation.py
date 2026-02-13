"""
Estimate sequence length distribution for FIMX-style dataset (including answer/explanation injection).
This uses the same preprocessing path as training: LazySupervisedDatasetForFIMX.
"""

import numpy as np
import torch
from types import SimpleNamespace
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from llava.train.train import (
    DataArguments,
    LazySupervisedDatasetForFIMX,
    DataCollatorForSupervisedDataset,
)
from llava import conversation as conversation_lib


def main():
    path = "/home/20223206/ACT-X/actX_train_llada.json"
    image_folder = "/home/20223206/ACT-X"
    tok = AutoTokenizer.from_pretrained("/home/20223206/model/LLaDA-V-HF", use_fast=False)

    # Match training prompt template
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_llada"].copy()

    args = DataArguments(data_path=path, image_folder=image_folder)
    ds = LazySupervisedDatasetForFIMX(
        data_path=path,
        tokenizer=tok,
        data_args=args,
        answer_block_size=20,
    )
    # We only need text lengths; bypass image loading with a tiny tensor.
    ds._process_image = lambda *a, **k: (torch.zeros(1, 3, 1, 1), (0, 0), "image")

    collator = DataCollatorForSupervisedDataset(
        tokenizer=tok, training_args=SimpleNamespace(use_conversation_mask=False)
    )

    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=collator)

    lengths = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    for batch in tqdm(loader, desc="tokenizing (FIMX path)"):
        labels = batch["labels"]
        # Length per sample = non-pad tokens (pad_id used for LLaDA mode).
        lens = (labels != pad_id).sum(dim=-1).tolist()
        lengths.extend(lens)

    lengths = np.array(lengths)
    for p in [50, 90, 95, 99, 100]:
        print(f"p{p}: {np.percentile(lengths, p)}")
    print(f"mean: {lengths.mean():.1f}, max: {lengths.max()}, min: {lengths.min()}")


if __name__ == "__main__":
    main()
