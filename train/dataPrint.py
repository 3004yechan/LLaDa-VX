import json
path = "/home/20223206/ACT-X/actX_train_llada.json"
data = json.load(open(path))
bad = data[0]  # id 020934932
print(bad["conversations"])
print("answer:", bad.get("answer"))
print("explanation:", bad.get("explanation"))


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/20223206/model/LLaDA-V-HF", use_fast=False)
print("reserved id:", tok.convert_tokens_to_ids("<|reserved_token_1|>"))