import pandas as pd
import json

INPUT_PATH = "../Ulysses-RFCorpus/bills_dataset.csv"
OUTPUT_PATH = "data/ulysses_sample.json"

df = pd.read_csv(INPUT_PATH)

sample = df.head(30)

data = []

for _, row in sample.iterrows():
    text = row.get("text")
    summary = row.get("txt_ementa")

    if isinstance(text, str) and isinstance(summary, str):
        data.append({
            "text": text,
            "summary": summary
        })

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"Dataset convertido com {len(data)} exemplos.")