from modal import Image, Stub, method
from pprint import pformat
from typing import Generator

MODEL_URL = "https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/resolve/main/model-q4_K.gguf"
MODEL_DIR = "/root/models_dir"
MODEL_NAME = "saiga_mistral7B-q4_K.gguf"

stub = Stub("modal_saiga_mistral_simple")

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "llama-cpp-python",
        "langchain"
    ).apt_install(
        "wget"
    ).run_commands(
        f"mkdir -p {MODEL_DIR}",
        f"cd {MODEL_DIR}",
        f"wget {MODEL_URL} -O {MODEL_DIR}/{MODEL_NAME}",
    )
)

# set stub's image to the one we've initialized
stub.image = image
with stub.image.run_inside():
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


def inject_role(
    model: Llama,
    role_str: str,
    content: str
) -> list[int]:
    msg_tokens = model.tokenize(content.encode("utf-8"))
    msg_tokens.insert(1, ROLE_TOKENS[role_str])
    msg_tokens.insert(2, LINE_BREAK_TOKEN)
    msg_tokens.append(model.token_eos())
    return msg_tokens


def add_setup_tokens(model: Llama) -> None:
    # - get system tokens
    setup_tokens = inject_role(model, "system", SYSTEM_PROMPT)
    # - for now tokens only contains setup (aka "system") tokens
    # - give these tokens to the model
    # (basically specific setup prompt tokenized by model)
    model.eval(setup_tokens)
    print("System prompt evaluated")


def prepare_prompt_tokens(model: Llama, prompt: str) -> list[int]:
    prompt_tokenized = inject_role(model, "user", prompt)
    # inject more specific tokens AGAIN
    return prompt_tokenized + [
        model.token_bos(),
        BOT_TOKEN,
        LINE_BREAK_TOKEN
    ]


def detokenize_generated(
    model: Llama,
    generator: Generator[int, None, None],
    tokens: list[int]
) -> list[str]:
    res = ""
    # TODO: maybe optimize it?
    for token in generator:
        token_str = model.detokenize(
            [token]
        ).decode(
            encoding="utf-8",
            errors="ignore"
        )

        # save generated tokens
        tokens.append(token)

        # do not add "end of string" tokens
        if token == model.token_eos():
            break

        res += token_str
    return res


@stub.cls(gpu="A10G")
class SaigaMistral7B:
    def __enter__(self):
        from llama_cpp import Llama
        # set path to GGUF model file which we downloaded earlier
        self.model_path = f"{MODEL_DIR}/{MODEL_NAME}"

        # token list serves as conversation memory
        self.tokens = []

        # initialize llama.cpp compatible model
        self.model = Llama(
            self.model_path,
            n_ctx=2000,
            n_parts=1,
        )

        # add tokenized initial "system" prompt
        # - which is specific for this model
        add_setup_tokens(self.model)

    @method()
    def run_prompt(self, prompt: str, logging_on: bool = True) -> dict[str, str]:
        self.tokens += prepare_prompt_tokens(self.model, prompt)
        if logging_on:
            # pformat is a method that formats python object beautifully
            # (it's a part of built-in pprint module, so no dependencies here)
            print(
                "Tokens (including user prompt + role tokens)",
                pformat(self.tokens)
            )

        if logging_on:
            print("Full PROMPT is:", self.model.detokenize(self.tokens).decode(
                "utf-8",
                errors="ignore"
            ))

        # generation step = actual use of LLM
        generator = self.model.generate(
            self.tokens,
            top_k=30,
            top_p=0.9,
            temp=0.2,
            repeat_penalty=1.1
        )

        return {
            "user_prompt": prompt,
            "llm_response": detokenize_generated(self.model, generator, self.tokens),
        }

    @method()
    def decode_tokens(self) -> str:
        res = ""
        for token in self.tokens:
            res += self.model.detokenize([token]).decode(
                "utf-8",
                errors="ignore"
            )

        return res


@stub.local_entrypoint()
def entrypoint(logging: bool):
    prompts = [
        "Кто ты?",
        "Почему трава зелёная",
        "Что я спросил у тебя до этого",
        "Do you speak English?"
    ]

    answers = []
    saiga = SaigaMistral7B()
    for prompt in prompts:
        resp = saiga.run_prompt.remote(prompt, logging_on=logging)
        answers.append(resp)

    print(saiga.decode_tokens.remote())
    for answer in answers:
        print(answer)
