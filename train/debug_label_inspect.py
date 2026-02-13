from transformers import AutoTokenizer
from llava.train.train import DataArguments, LazySupervisedDatasetForFIMX
from llava import conversation as conversation_lib
import torch

path = "/home/20223206/ACT-X/actX_train_llada.json"
img_folder = "/home/20223206/ACT-X"
tok = AutoTokenizer.from_pretrained("/home/20223206/model/LLaDA-V-HF", use_fast=False,local_files_only=True)
conversation_lib.default_conversation = conversation_lib.conv_templates["llava_llada"].copy()

ds = LazySupervisedDatasetForFIMX(
    data_path=path,
    tokenizer=tok,
    data_args=DataArguments(data_path=path, image_folder=img_folder),
    answer_block_size=20,
)

# Skip real image load
ds._process_image = lambda *a, **k: (torch.zeros(1,3,1,1), (0,0), "image")

s = ds[0]
valid = s["labels"][s["labels"] != -100]
print("valid tokens:", valid.tolist())
print("decoded:", tok.decode(valid.tolist(), skip_special_tokens=True,
clean_up_tokenization_spaces=False))