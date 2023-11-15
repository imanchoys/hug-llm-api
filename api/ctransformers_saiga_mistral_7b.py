from typing import Any, Generator
from ctransformers import AutoModelForCausalLM, LLM
from rich.pretty import pretty_repr
from loguru import logger

# Set gpu_layers to the number of layers to offload to GPU.
# Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(
    "IlyaGusev/saiga_mistral_7b_gguf",
    model_file="model-q4_K.gguf",
    model_type="llama",
    gpu_layers=50
    # gpu_layers=100
)

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


def pretty_log(obj: Any, debug: bool = False) -> None:
    msg = pretty_repr(obj)
    if debug:
        logger.debug(msg)
        return

    logger.info(msg)


def get_message_token(
    llm: LLM,
    role_id: str = "system",
    content: str = SYSTEM_PROMPT
) -> str:
    # TODO: customize "system tokens" creation
    msg_tokens = llm.tokenize(content)
    msg_tokens.insert(1, ROLE_TOKENS[role_id])
    msg_tokens.insert(2, LINE_BREAK_TOKEN)
    msg_tokens.append(llm.eos_token_id)
    return msg_tokens


def detokenize_results(
    llm: LLM,
    tokens: list[int],
    generator: Generator[int, None, None]
) -> list[str]:
    res = []
    for token in generator:
        token_str = llm.detokenize(tokens=[token], decode=True)

        tokens.append(token)
        if llm.is_eos_token(token):
            break
        res.append(token_str)

    return res


def run(
    model: LLM,
    prompt: str,
    tokens: list[int],
    print_tokens: bool = True
):
    # save system tokens to session
    sys_tokens = get_message_token(model)
    # for now tokens is equal to system tokens
    # so we save them as well
    tokens = sys_tokens
    # give the system tokens to model
    model.eval(tokens)

    message_tokens = get_message_token(
        llm=model,
        role_id="user",
        content=prompt
    )

    # tokenization step
    role_tokens = [
        model.bos_token_id,
        BOT_TOKEN,
        LINE_BREAK_TOKEN
    ]

    tokens += message_tokens + role_tokens
    if print_tokens:
        logger.info(pretty_repr(tokens))

    full_prompt = model.detokenize(tokens)
    if print_tokens:
        logger.info(pretty_repr(model.tokenize(full_prompt)))

    # generation step = actual use of LLM
    generator = model.generate(
        tokens,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repetition_penalty=1.1
    )

    res = detokenize_results(
        model,
        tokens,
        generator
    )

    logger.info(pretty_repr(res))

    answer = "".join(res)
    return {
        "prompt": prompt,
        "result": answer,
    }


# answer = llm("AI is going to")
print(run(model=llm, prompt="Почему трава зелёная", tokens=[]))
