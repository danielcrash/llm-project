from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                  use_stemmer=True)


def is_truncated(text: str) -> bool:
    return text.strip().endswith(("...", "…")) or len(text.split()) < 5


def detect_issue(scores: dict, generated: str):
    if is_truncated(generated):
        return "truncation"

    if scores["rougeL"].fmeasure < 0.7:
        return "low_overlap"

    return None


def evaluate(generated, reference):
    scores = scorer.score(reference, generated)

    issue = detect_issue(scores, generated)

    if issue:
        print(f"⚠️ {issue}: {generated[:120]}")

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }