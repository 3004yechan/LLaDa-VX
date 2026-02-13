import torch
from llava.model.builder import load_pretrained_model

pretrained = "./exp/llada_v_qlora_single"
model_base = "/home/20223206/model/LLaDA-V-HF"
model_name = "llava_llada_lora"

# 모델 로드 (merge 여부와 무관하게 shape만 본다)
tok, model, image_processor, _ = load_pretrained_model(
    pretrained, model_base, model_name,
    attn_implementation="sdpa", device_map="cuda:0",
    load_4bit=False,
)

# 기대 shape
mm = model.get_model()
proj = mm.mm_projector.state_dict()
print("Expected mm_projector shapes:")
for k,v in proj.items():
    print(f"{k}: {tuple(v.shape)}")

# non_lora_trainables 내용을 확인
state = torch.load(f"{pretrained}/non_lora_trainables.bin",
map_location="cpu")
print("\nnon_lora_trainables keys:")
for k,v in state.items():
    if "mm_projector" in k:
        print(f"{k}: {tuple(v.shape)}")