from transformers.generation import stopping_criteria
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from dataclasses import asdict
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook

from PIL import Image
import requests
import copy
import torch
import time

import sys
import warnings

prompt_interval_steps = 25
gen_interval_steps = 7
transfer_ratio = 0.25

# fast dllm 적용시 3초(양자화X기준), 미적용시 메모리 부족
# 중요!! llada-V 코드에서 중간중간 64비트로 업캐스팅하는 코드를 모두 삭제했음 품질 향상이 이유였던거 같은데 성능괜찮은거 보니까 이정도는 trade-off 가능할듯
use_fast_dllm = False  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")
pretrained = "GSAI-ML/LLaDA-V"
# pretrained = "./exp/llada_v_qlora_single"
# model_base = "/home/20223206/model/LLaDA-V-HF"

model_name = "llava_llada_lora"
device = "cuda:0"
device_map = "cuda:0"
# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=device_map, load_8bit=True)  # Add any other thing you want to pass in llava_model_args
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, attn_implementation="sdpa", device_map=device_map, load_4bit=True)  # Add any other thing you want to pass in llava_model_args

model.eval()
image = Image.open("tennis.jpg")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llada"
# question = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."
question = DEFAULT_IMAGE_TOKEN + "\nWhat activity is the person (or people) performing in this image? Before you answer, please explain the reason first using the format below: 'Because..., the answer is...'. You can use <|reserved_token_1|> to trigger an early stop for a brief explanation."

explanation_max_token = 70 
answer_max_token = 12
FIM = '<|reserved_token_1|>'
# FIM = ''
draft_answer = f'''Because{"<|mdm_mask|>"*explanation_max_token}, the answer is{"<|mdm_mask|>"*answer_max_token}<|eot_id|>'''

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

model.eval()
if use_fast_dllm:
    register_fast_dllm_hook(model)
    print("Testing with Fast dLLM hook enabled")
elif use_dllm_cache:
    dLLMCache.new_instance(
        **asdict(
            dLLMCacheConfig(
                prompt_interval_steps=prompt_interval_steps,
                gen_interval_steps=gen_interval_steps,
                transfer_ratio=transfer_ratio,
            )
        )
    )
    register_cache_LLaDA_V(model, "model.layers")
    print("Testing with cache enabled")
else:
    print("Testing without cache")

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

# draft_tokens = tokenizer(draft_answer,return_tensors='pt').input_ids.to(device) 
draft_tokens = tokenizer(draft_answer,return_tensors='pt').to(input_ids.device).input_ids

start_time = time.time()
cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    steps=128, gen_length=128, block_length=128, tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'], 
    prefix_refresh_interval=32,
    threshold=1,
    draft_tokens=draft_tokens,
)
end_time = time.time()
generation_time = end_time - start_time
print(f"Generation time: {generation_time:.4f} seconds")

print(cont)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
print(text_outputs)

# import re

# def _capitalize_first(text: str) -> str:
#     for idx, ch in enumerate(text):
#         if ch.isalpha():
#             return f"{text[:idx]}{ch.upper()}{text[idx+1:]}"
#     return text

# def _clean_special_tokens(text: str) -> str:
#     text = text.replace("<|reserved_token_1|>", " ")
#     text = text.replace("<|eot_id|>", " ")
#     return re.sub(r"\s+", " ", text).strip()

# print("-" * 70)
# token_limit = explanation_max_token + 1
# token_pattern = re.compile(r"\S+")

# for t in text_outputs:
#     cleaned = _clean_special_tokens(t)
#     explanation_slice_end = None
#     for idx, match in enumerate(token_pattern.finditer(cleaned), start=1):
#         if idx == token_limit:
#             explanation_slice_end = match.end()
#             break
#     if explanation_slice_end is None:
#         explanation_slice_end = len(cleaned)

#     explanation_raw = cleaned[:explanation_slice_end].strip()
#     answer_raw = cleaned[explanation_slice_end:].strip()

#     explanation = re.sub(r"<\|[^>]+?\|>", "", explanation_raw).strip()
#     explanation = _capitalize_first(explanation)
#     if (
#         explanation.endswith(",")
#         and re.match(r"^\s*,?\s*the answer is", answer_raw, flags=re.IGNORECASE)
#     ):
#         explanation = explanation[:-1].rstrip()
#     if explanation and explanation[-1] not in ".!?":
#         explanation = f"{explanation}."

#     answer = re.sub(r"<\|[^>]+?\|>", "", answer_raw).strip()
#     answer = re.sub(r"^,?\s*the answer is[:\s]*", "", answer, flags=re.IGNORECASE)
#     answer = _capitalize_first(answer.strip())

#     print("Answer:", answer, "\n")
#     print("Explanation:", explanation)
#     print("-" * 70)
