# Credits to the creators of Saiga Mistral 7b model (cc-by-4.0): https://huggingface.co/IlyaGusev/saiga_mistral_7b_lora
# > Saiga Mistral 7b is itself based on Mistral OpenOrca (cc-by-4.0): https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca
# Llama.cpp version (apache-2.0): https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf 

# This demo works with llama.cpp python bindings >= 0.1.79 - it expects model GGUF format:
#   - to install: `pip install llama-cpp-python`
from llama_cpp import Llama

# Saiga-Mistral-7b specific format tokens 
SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
SYSTEM_TOKEN = 1587
USER_TOKEN = 2188
BOT_TOKEN = 12435
LINE_BREAK_TOKEN = 13

# WARNING: to run this demo you must download (or pre-download before running)
# the ".gguf" model file from huggingface:
#   - https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q2_K.gguf (~3.08 GB)
#   - https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q4_K.gguf (~4.37 GB)
#   - https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q8_0.gguf (~7.7 GB)
# > 10GB RAM for q8_0 and less for smaller quantizations

# to pre-download simply run:
# ```bash
# wget https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q4_K.gguf
# ```

# path for .gguf model file (could be either run on CPUs or GPUs)
PATH_TO_GGUF = "./models_dir/saiga_mistral_7b/model-q4_K.gguf"

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


def interact(
    model_path: str,
    n_ctx=2000,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1,
    print_tokens=True
):
    model = Llama(
        model_path,
        n_ctx=n_ctx,
        n_parts=1,
    )

    system_tokens = get_system_tokens(model)
    tokens = system_tokens
    model.eval(tokens)

    while True:
        # get prompt from user input
        user_message = input("User: ")

        # create tokens compatible with this model
        # (just manually inserting a few constants defined earlier, like:
        # USER_TOKEN, BOT_TOKEN, SYSTEM_TOKEN, SYSTEM_PROMPT)
        message_tokens = get_message_tokens(
            model=model,
            role="user",
            content=user_message
        )

        # tokenization step
        role_tokens = [model.token_bos(), BOT_TOKEN, LINE_BREAK_TOKEN]
        tokens += message_tokens + role_tokens
        if print_tokens:
            print(tokens)

        full_prompt = model.detokenize(tokens)
        if print_tokens:
            print(model.tokenize(full_prompt))

        # generation step = actual use LLM
        generator = model.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            repeat_penalty=repeat_penalty
        )

        for token in generator:
            token_str = model.detokenize([token]).decode(
                "utf-8",
                errors="ignore"
            )

            tokens.append(token)
            if token == model.token_eos():
                break
            print(token_str, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    # check if model is downloaded to run
    interact(model_path=PATH_TO_GGUF, print_tokens=False)
