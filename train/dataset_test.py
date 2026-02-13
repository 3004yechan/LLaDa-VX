from llava.train.train import LazySupervisedDatasetForFIMX, DataArguments
from transformers import AutoTokenizer

model_dir = "/home/20223206/model/LLaDA-V-HF"
data_path = "/home/20223206/ACT-X/actX_train_llada.json"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

# DataArguments는 최소한 image_folder만 맞춰 주면 됩니다(이미지 안 쓸 때도 필수 필드라 채워야 함).
data_args = DataArguments(
    data_path=data_path,
    image_folder="/home/20223206/ACT-X/images",
    image_aspect_ratio="anyres_max_4",
)

ds = LazySupervisedDatasetForFIMX(
    data_path=data_path,
    tokenizer=tokenizer,
    data_args=data_args,
    answer_block_size=20,   # 스크립트에서 설정한 값 확인
)

raw = ds.list_data_dict[0]   # 첫 샘플 원본
label_ids = ds._build_label_ids(raw)
label_text = tokenizer.decode(label_ids, skip_special_tokens=False,
clean_up_tokenization_spaces=False)

print("id:", raw.get("id", 0))
print("human turn:", raw["conversations"][0]["value"])
print("answer field:", raw.get("answer") or raw.get("answers"))
print("explanation field:", raw.get("explanation"))
print("constructed label_text:", label_text)
print("constructed label_ids:", label_ids[:60])