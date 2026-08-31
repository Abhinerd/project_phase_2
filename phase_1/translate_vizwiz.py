import json
import argostranslate.translate
from tqdm import tqdm

# Load JSON
with open("train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

translated_data = []

for item in tqdm(data):
    question_en = item["question"]

    # translate question
    question_hi = argostranslate.translate.translate(
        question_en, "en", "hi"
    )

    item["question_hi"] = question_hi
    translated_data.append(item)

# Save new dataset
with open("vizwiz_train_hindi.json", "w", encoding="utf-8") as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)

print("Translation complete!")