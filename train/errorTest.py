from llava.train.train import LazySupervisedDatasetForFIMX, DataArguments
from transformers import AutoTokenizer
import torch, json

from llava.model.builder import load_pretrained_model

LLM = "/home/20223206/model/LLaDA-V-HF"
VISION = "model/siglip2-so400m-patch14-384"

# Load tokenizer + vision processor. Avoid passing vision_tower kw (HF checkpoint doesn't expect it).
tokenizer, model, image_processor, _ = load_pretrained_model(
    LLM,
    None,
    "llava_llada",
    load_4bit=False,
    device_map="cpu",  # CPU로만 확인할 때 충분
    attn_implementation="sdpa",
)

data_args = DataArguments(
    data_path="/home/20223206/ACT-X/actX_train_llada.json",
    image_folder="/home/20223206/ACT-X",
    image_aspect_ratio="anyres_max_4",
    image_grid_pinpoints="(1x1),...,(6x6)",
)
data_args.image_processor = image_processor  # attach processor for image loading

ds = LazySupervisedDatasetForFIMX(data_args.data_path, tokenizer, data_args, answer_block_size=20)
vocab = tokenizer.vocab_size
for i in range(len(ds)):
    ex = ds[i]
    ids = ex["input_ids"]
    # Some preprocess paths return a list of tensors (e.g., for vision/text mix). Stack if needed.
    if isinstance(ids, list):
        ids = torch.stack(ids)
    if ids.max() >= vocab or ids.min() < 0:
        print(f"bad sample idx={i}, id={ex.get('id')}, min={ids.min().item()}, max={ids.max().item()}")
        break
