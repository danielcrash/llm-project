from src.ollama_client import generate


def summarize(text):
    prompt = f"""
Você é um assistente especializado em análise jurídica.

Sua tarefa é gerar um resumo extremamente fiel ao texto original.

Regras:
- Use no máximo 1 frase.
- Preserve o vocabulário original sempre que possível.
- Evite sinônimos e paráfrases desnecessárias.
- Não adicione explicações ou interpretações.

Texto:
{text}

Resumo:
"""
    return generate(prompt)
