from src.summarizer import summarize
from src.evaluator import evaluate
import json
import os
from datetime import datetime
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def compute_embedding_similarity(a, b):
    emb = embedding_model.encode([a, b])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


KEY_TERMS = {
    "obrigacao": ["deverá", "fica obrigado", "obrigatório"],
    "permissao": ["poderá", "faculta", "autoriza"],
    "proibicao": ["vedado", "proibido"]
}


def classify_modality(text):
    t = text.lower()
    for label, terms in KEY_TERMS.items():
        if any(term in t for term in terms):
            return label
    return "neutro"


def detect_semantic_shift(ref, pred):
    return classify_modality(ref) != classify_modality(pred)


def clean_text(text: str) -> str:
    # remove URLs
    text = re.sub(r'http\S+', ' ', text)

    text = re.sub(r'CD\d+', ' ', text)

    lines = []
    for line in text.splitlines():
        if any(k.lower() in line.lower() for k in [
            "assinado eletronicamente",
            "câmara dos deputados",
            "gabinete",
            "documento gerado",
            "autenticidade",
        ]):
            continue
        lines.append(line)
    text = " ".join(lines)

    text = re.sub(r'\s+', ' ', text).strip()

    return text[:4000]


def extract_main_section(text: str) -> str:
    if "Justificação" in text:
        text = text.split("Justificação")[0]

    if "INDICAÇÃO" in text and "REQUERIMENTO" in text:
        text = text.split("REQUERIMENTO")[0]

    return text

def run_pipeline(input_path, output_path=None):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items")

    results = []
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/results_{timestamp}.json"

    for i, item in enumerate(data):
        raw_texto = item["text"]
        texto = clean_text(raw_texto)
        texto = extract_main_section(texto)
        if not texto or len(texto.strip()) < 20:
            continue
        referencia = item.get("summary", "")

        resumo = summarize(texto)
        if any(x in resumo for x in ["I'm ready", "I apologize"]):
            print("⚠️ resposta inválida:", resumo[:120])
            resumo = texto.split(".")[0]

        # fallback if summary is too short
        if len(resumo.split()) < 5:
            resumo = texto.split(".")[0]

        scores = evaluate(resumo, referencia)
        embedding_sim = compute_embedding_similarity(referencia, resumo)
        semantic_shift = detect_semantic_shift(referencia, resumo)

        # --- Composite scoring (less brittle than binary thresholds) ---
        rouge1 = float(scores.get("rouge1", 0.0))
        rougeL = float(scores.get("rougeL", 0.0))
        emb = float(embedding_sim)

        # length control (avoid too short/long summaries)
        length_ratio = len(resumo) / max(len(referencia), 1)
        length_penalty = abs(1 - length_ratio)

        # weighted score (semantic > lexical for legal domain)
        final_score = (0.3 * rouge1) + (0.2 * rougeL) + (0.5 * emb)
        final_score -= 0.1 * length_penalty

        flags = []

        if rougeL < 0.45:
            flags.append("low_lexical")

        if emb < 0.65:
            flags.append("semantic_divergence")

        if semantic_shift:
            flags.append("possible_legal_shift")

        for f in flags:
            print(f"{f}: {resumo[:120]}")

        if final_score >= 0.7:
            status = "ok"
        elif final_score >= 0.55:
            status = "review"
        else:
            print(f"4fallback acionado | score={final_score:.2f} | rougeL={rougeL:.2f} | emb={emb:.2f}")
            resumo = texto.split(".")[0]
            status = "fallback"

        print(
            f"[{i + 1}/{len(data)}] processed | score={final_score:.2f} | rougeL={rougeL:.2f} | emb={emb:.2f} | status={status}"
        )

        results.append({
            "input": texto,
            "generated": resumo,
            "reference": referencia,
            "scores": {
                "rouge1": float(scores.get("rouge1", 0.0)),
                "rouge2": float(scores.get("rouge2", 0.0)),
                "rougeL": float(scores.get("rougeL", 0.0)),
            },
            "embedding": float(emb),
            "final_score": float(final_score),
            "flags": flags,
            "status": status
        })

    # Calculate average ROUGE scores
    if results:
        avg_rouge1 = sum(r["scores"]["rouge1"] for r in results) / len(results)
        avg_rouge2 = sum(r["scores"]["rouge2"] for r in results) / len(results)
        avg_rougeL = sum(r["scores"]["rougeL"] for r in results) / len(results)

        print("\n=== MÉDIAS ===")
        print(f"ROUGE-1: {avg_rouge1:.4f}")
        print(f"ROUGE-2: {avg_rouge2:.4f}")
        print(f"ROUGE-L: {avg_rougeL:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
