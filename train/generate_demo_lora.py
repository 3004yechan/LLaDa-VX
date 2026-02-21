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

prompt_interval_steps = 8
gen_interval_steps = 2
transfer_ratio = 0.8
use_fast_dllm = False  # using fast-dLLM (https://github.com/NVlabs/Fast-dLLM) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 6s to generate 128 tokens.
use_dllm_cache = False  # using dLLM-Cache(https://github.com/maomaocun/dLLM-cache) to speed up generation. Set to True to enable caching or False to test without it. In A100, it uses around 25s to generate 128 tokens.

warnings.filterwarnings("ignore")
# pretrained = "GSAI-ML/LLaDA-V"
pretrained = "/home/20223206/exp/llada_v_qlora_actx_single"
model_base = "/home/20223206/model/LLaDA-V-HF"

model_name = "llava_llada_lora"
device = "cuda:0"
device_map = "cuda:0"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, model_base, model_name, attn_implementation="sdpa", device_map=device_map, load_4bit=True)  # Add any other thing you want to pass in llava_model_args

model.eval()
hook_model = model.get_base_model() if hasattr(model, "get_base_model") else model
# image = Image.open("/workspace/ACT-X/images/026558760.jpg")
image = Image.open("/home/20223206/ACT-X/images/071674638.jpg")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "llava_llada"
question = DEFAULT_IMAGE_TOKEN + "\nWhat activity is the person (or people) performing in this image? Before you answer, please explain the reason first using the format below: 'Because..., the answer is...'. You can use <|reserved_token_1|> to trigger an early stop for a brief explanation."
# question = DEFAULT_IMAGE_TOKEN + "\nWhat she Doing? Explain why."

explanation_max_token = 70 
answer_max_token = 30 
PAD = '<|reserved_token_1|>'
enable_reserved_collapse = True
reserved_token_id = 126085
mdm_mask_id = 126336
enable_attention_remask = True
attention_remask_top_tokens = 32
attention_remask_top_heads = 8
attention_remask_low_precision_softmax = True
# FIM = ''
draft_answer = f'''Because{"<|mdm_mask|>"*explanation_max_token}, the answer is{"<|mdm_mask|>"*answer_max_token}<|eot_id|>'''

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

model.eval()
if use_fast_dllm:
    register_fast_dllm_hook(hook_model)
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
    register_cache_LLaDA_V(hook_model, "model.layers")
    print("Testing with cache enabled")
else:
    print("Testing without cache")

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


draft_tokens = tokenizer(draft_answer,return_tensors='pt').to(input_ids.device).input_ids

start_time = time.time()
# LLaDA generate_with_embeds expects stop strings (it tokenizes them internally).
stop_tokens = ["<|eot_id|>"]

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    steps=64,
    gen_length=128,
    block_length=128,
    tokenizer=tokenizer,
    stopping_criteria=stop_tokens,
    prefix_refresh_interval=32,
    threshold=1, # for fast-dllm. default=1. None일경우 정해진 step을 모두 채움.
    draft_tokens=draft_tokens,
    enable_reserved_collapse=enable_reserved_collapse,
    reserved_token_id=reserved_token_id,
    mdm_mask_id=mdm_mask_id,
    enable_attention_remask=enable_attention_remask,
    attention_remask_top_tokens=attention_remask_top_tokens,
    attention_remask_top_heads=attention_remask_top_heads,
    attention_remask_low_precision_softmax=attention_remask_low_precision_softmax,
)
end_time = time.time()
generation_time = end_time - start_time
print(f"Generation time: {generation_time:.4f} seconds")

print(cont)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
print(text_outputs)

RESERVED_TOKEN_1_ID = 126085
cont_without_reserved = [
    [token_id for token_id in seq.tolist() if token_id != RESERVED_TOKEN_1_ID]
    for seq in cont
]
text_outputs_without_reserved = [
    tokenizer.decode(seq, skip_special_tokens=False)
    for seq in cont_without_reserved
]
print("result:", text_outputs_without_reserved)
