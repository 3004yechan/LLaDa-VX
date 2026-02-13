import json
src = "./actX_train_filled_llada.json"
dst = "./actX_train_filled_llada_trim.json"
data = json.load(open(src, encoding="utf-8"))
cleaned = [{"id": s["id"], "image": s["image"], "conversations": s["conversations"]} for s in data]
json.dump(cleaned, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"{len(cleaned)} samples -> {dst}")