from typing import Generator, Sequence
from llama_cpp import Llama

# Saiga-Mistral-7b specific format tokens
SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1587
USER_TOKEN = 2188
BOT_TOKEN = 12435
LINE_BREAK_TOKEN = 13
ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}


def get_message_tokens(model: Llama, role: str, content: str) -> list[int]:
    message_tokens = model.tokenize(content.encode("utf-8"))
    message_tokens.insert(1, ROLE_TOKENS[role])
    message_tokens.insert(2, LINE_BREAK_TOKEN)
    message_tokens.append(model.token_eos())
    return message_tokens


def get_system_tokens(model: Llama):
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
    return get_message_tokens(model, **system_message)


def detokenize_results(
    model: Llama,
    tokens: list[int],
    generator: Generator[int, Sequence[int] | None, None]
) -> list[str]:
    res = []
    for token in generator:
        token_str = model.detokenize([token]).decode(
            "utf-8",
            errors="ignore"
        )

        tokens.append(token)
        if token == model.token_eos():
            break
        res.append(token_str)
        
    return res
