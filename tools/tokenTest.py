import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-V")

text= "The answer is sitting<|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|><|reserved_token_1|>"

encoding = tokenizer(text, add_special_tokens=True, return_attention_mask=False)
token_len = len(encoding["input_ids"])

print(token_len)
print(encoding)

