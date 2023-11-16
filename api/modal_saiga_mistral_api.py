from modal import Image, Stub, method
from pprint import pformat
from typing import Generator

stub = Stub("saiga_mistral_ctransformers")


def download_model():
    from huggingface_hub import hf_hub_download
    logger.info("Downloading model - only if do not exist in FS cache")

    hf_hub_download(
        "IlyaGusev/saiga_mistral_7b_gguf",
        "model-q4_K.gguf",
        repo_type="model"
    )

    logger.info("HuggingFace Hub downloading step finished!")


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "huggingface-hub",
        "langchain",
        "loguru"
    )
    .run_commands("CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers")
    .run_function(download_model)
)

# set stub's image to the one we've initialized
stub.image = image
with stub.image.run_inside():
    from loguru import logger
    from ctransformers import AutoModelForCausalLM, LLM

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


def add_setup_tokens(model: LLM) -> None:
    # - get system tokens
    setup_tokens = inject_role(model, "system", SYSTEM_PROMPT)
    # - for now tokens only contains setup (aka "system") tokens
    # - give these tokens to the model
    # (basically specific setup prompt tokenized by model)
    model.eval(setup_tokens)
    logger.info("Initial system prompt evaluated")


def prepare_prompt_tokens(model: LLM, prompt: str) -> list[int]:
    prompt_tokenized = inject_role(model, "user", prompt)
    # inject more specific tokens AGAIN
    return prompt_tokenized + [
        model.bos_token_id,
        BOT_TOKEN,
        LINE_BREAK_TOKEN
    ]


def detokenize_generated(
    model: LLM,
    generator: Generator[int, None, None],
    tokens: list[int]
) -> list[str]:
    res = ""
    # TODO: maybe optimize it?
    for token in generator:
        token_str = model.detokenize([token], decode=True)

        # save generated tokens
        tokens.append(token)

        # do not add "end of string" tokens
        if model.is_eos_token(token):
            break

        res += token_str
    return res


@stub.cls(gpu="A10G")
class SaigaMistral7B:
    def __enter__(self):
        # token list serves as conversation memory
        self.tokens = []
        
        # Set gpu_layers to the number of layers to offload to GPU.
        # Set to 0 if no GPU acceleration is available on your system.
        self.model = AutoModelForCausalLM.from_pretrained(
            "IlyaGusev/saiga_mistral_7b_gguf",
            model_file="model-q4_K.gguf",
            model_type="llama",
            local_files_only=True,
            gpu_layers=50
            # gpu_layers=100
        )
        # add tokenized initial "system" prompt
        # - which is specific for this model
        setup_tokens = inject_role(self.model, "system", SYSTEM_PROMPT)
        # - for now tokens only contains setup (aka "system") tokens
        # - give these tokens to the model
        # (basically specific setup prompt tokenized by model)
        self.model.eval(setup_tokens)
        logger.info("Finished initial model setup")

    @method()
    def run_prompt(self, prompt: str, logging_on: bool = True) -> dict[str, str]:
        self.tokens += prepare_prompt_tokens(self.model, prompt)
        if logging_on:
            # pformat is a method that formats python object beautifully
            # (it's a part of built-in pprint module, so no dependencies here)
            logger.info(
                "Tokens (including user prompt + role tokens)",
                pformat(self.tokens)
            )

        if logging_on:
            logger.info(
                "Full PROMPT is:",
                self.model.detokenize(self.tokens, decode=True)
            )

        # generation step = actual use of LLM
        generator = self.model.generate(
            self.tokens,
            top_k=30,
            top_p=0.9,
            temperature=0.2,
            repetition_penalty=1.1
        )

        return {
            "user_prompt": prompt,
            "llm_response": detokenize_generated(self.model, generator, self.tokens),
        }

    @method()
    def decode_tokens(self) -> str:
        res = ""
        for token in self.tokens:
            res += self.model.detokenize([token], decode=True)

        return res


@stub.local_entrypoint()
def entrypoint(logging: bool = False):
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

    logger.success(saiga.decode_tokens.remote())
    for answer in answers:
        logger.success(answer)
