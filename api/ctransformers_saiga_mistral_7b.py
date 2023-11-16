from typing import Generator
from pprint import pformat

from ctransformers import AutoModelForCausalLM, LLM
from huggingface_hub import hf_hub_download

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


def inject_role(
    llm: LLM,
    role_str: str,
    content: str
) -> list[int]:
    msg_tokens = llm.tokenize(content)
    msg_tokens.insert(1, ROLE_TOKENS[role_str])
    msg_tokens.insert(2, LINE_BREAK_TOKEN)
    msg_tokens.append(llm.eos_token_id)
    return msg_tokens


def add_setup_tokens(llm: LLM) -> None:
    # - get system tokens
    setup_tokens = inject_role(llm, "system", SYSTEM_PROMPT)
    # - for now tokens only contains setup (aka "system") tokens
    # - give these tokens to the model
    # (basically specific setup prompt tokenized by model)
    llm.eval(setup_tokens)
    print("System prompt evaluated")


def prepare_prompt_tokens(llm: LLM, prompt: str) -> list[int]:
    prompt_tokenized = inject_role(llm, "user", prompt)
    # inject more specific tokens AGAIN
    return prompt_tokenized + [
        llm.bos_token_id,
        BOT_TOKEN,
        LINE_BREAK_TOKEN
    ]


def detokenize_generated(
    llm: LLM,
    generator: Generator[int, None, None],
    token_queue: list[int]
) -> list[str]:
    res = []
    for token in generator:
        token_str = llm.detokenize(
            tokens=[token],
            decode=True
        )
        token_queue.append(token)
        if llm.is_eos_token(token):
            break
        res.append(token_str)

    return res


def run_prompt(
    model: LLM,
    prompt: str,
    tokens: list[int],
    logging_on: bool = True
):
    tokens += prepare_prompt_tokens(model, prompt)
    if logging_on:
        # pformat is a method that formats python object beautifully
        # (it's a part of built-in pprint module, so no dependencies here)
        print(
            "Tokens (including user prompt + role tokens)",
            pformat(tokens)
        )

    if logging_on:
        print("Full PROMPT is:", model.detokenize(tokens, decode="utf-8"))

    # generation step = actual use of LLM
    generator = model.generate(
        tokens,
        top_k=30,
        top_p=0.9,
        temperature=0.2,
        repetition_penalty=1.1
    )

    res = "".join(detokenize_generated(model, generator, tokens))
    return {
        "user_prompt": prompt,
        "llm_response": res,
    }


if __name__ == "__main__":
    print("Downloading model if do not exist in FS cache")
    hf_hub_download(
        "IlyaGusev/saiga_mistral_7b_gguf",
        "model-q4_K.gguf",
        repo_type="model"
    )
    
    print("HF Hub download finished!")
    # Set gpu_layers to the number of layers to offload to GPU.
    # Set to 0 if no GPU acceleration is available on your system.
    llm = AutoModelForCausalLM.from_pretrained(
        "IlyaGusev/saiga_mistral_7b_gguf",
        model_file="model-q4_K.gguf",
        model_type="llama",
        gpu_layers=50,
        local_files_only=True
        # gpu_layers=100
    )

    # add tokenized initial "system" prompt
    # - which is specific for this model
    add_setup_tokens(llm)

    prompts = [
        "Кто ты?",
        "Почему трава зелёная",
        "Что я спросил у тебя до этого",
        "Do you speak English?"
    ]

    tokens = []

    for prompt in prompts:
        resp = run_prompt(llm, prompt, tokens)
        print("Answer", resp["llm_response"])
