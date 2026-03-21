from src.summarizer import summarize
from src.evaluator import evaluate
import json
import os


def run_pipeline(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    results = []

    for item in data:
        texto = item["text"]
        referencia = item.get("summary", "")

        resumo = summarize(texto)

        scores = evaluate(resumo, referencia)

        results.append({
            "input": texto,
            "generated": resumo,
            "reference": referencia,
            "scores": scores
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)