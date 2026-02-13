from llava.model.builder import load_pretrained_model

# pretrained = "GSAI-ML/LLaDA-V"
pretrained = "./exp/llada_v_qlora_single"
model_base = "/home/20223206/model/LLaDA-V-HF"

model_name = "llava_llada_lora"
device = "cuda:0"
device_map = "cuda:0"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, model_base, model_name, attn_implementation="sdpa", device_map=device_map, load_4bit=True)  # Add any other thing you want to pass in llava_model_args

vocab_size = len(tokenizer)
embed_size = model.get_input_embeddings().weight.shape[0]
lm_head_size = model.get_output_embeddings().weight.shape[0]  # tied면 동일

print("vocab_size:", vocab_size)
print("input embedding size:", embed_size)
print("lm_head size:", lm_head_size)
print("matched:", vocab_size == embed_size == lm_head_size)