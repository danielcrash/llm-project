from src.ollama_client import generate


def summarize(text):
    prompt = f"""
Resuma o texto legislativo abaixo em UMA única frase.

Regras obrigatórias:
- Produza apenas UMA frase
- Máximo de 25 palavras
- Preserve o verbo principal (ex: "Sugere", "Requer", "Susta")
- Não explique
- Não adicione contexto
- Não complemente
- Use apenas a intenção principal da proposição
- Se houver múltiplos blocos, priorize o trecho que começa com "INDICAÇÃO" ou "Sugere"
- Seja o mais próximo possível do enunciado original

Texto:
{text}
"""
    return generate(prompt)
