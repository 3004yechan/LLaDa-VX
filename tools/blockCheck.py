import json
from transformers import AutoTokenizer

data_path = "./actX_train_filled_llada_trim.json"  # 검사할 파일
tok_path = "/home/20223206/model/LLaDA-V-HF"      # 사용한 토크나이저 경로/ID
target_block = 20
reserved = "<|reserved_token_1|>"

tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
rid = tokenizer.convert_tokens_to_ids(reserved)

bad = []
with open(data_path, encoding="utf-8") as f:
    samples = json.load(f)

for i, s in enumerate(samples):
    txt = s["conversations"][1]["value"]
    ids = tokenizer(txt, add_special_tokens=False).input_ids
    pos = [j for j, t in enumerate(ids) if t == rid]
    block_len = pos[-1] + 1 if pos else 0
    if block_len != target_block:
        bad.append((i, block_len, s.get("id", "")))

print(f"checked {len(samples)} samples")
if bad:
    for i, blen, sid in bad[:20]:
        print(f"[mismatch] idx={i} id={sid} block_len={blen}")
else:
    print("all good: reserved block length == 20")