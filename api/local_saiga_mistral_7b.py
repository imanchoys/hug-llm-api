import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding
)

MODEL_NAME = "IlyaGusev/saiga_mistral_7b"

# default constants are specific for Saiga-Mistral-7b model
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


class Conversation:
    def __init__(
        self,
        message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        response_template: str = DEFAULT_RESPONSE_TEMPLATE
    ):
        self.message_template = message_template
        self.response_template = response_template
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, _tokenizer) -> str:  # TODO: WHY tokenizer is not used?
        final_text = ""
        for msg in self.messages:
            msg_text = self.message_template.format(**msg)
            final_text += msg_text

        final_text += DEFAULT_RESPONSE_TEMPLATE
        return final_text.strip()


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompt: str,
    gen_config: GenerationConfig
) -> str:
    data: BatchEncoding = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    )

    # TODO: remove this
    for k, v in data.items():
        print("=> k:", type(k))
        print("=> k:", type(v))
        break

    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=gen_config)[0]
    
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


config = PeftConfig.from_pretrained(MODEL_NAME)
auto_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

peft_model = PeftModel.from_pretrained(
    auto_model,
    MODEL_NAME,
    torch_dtype=torch.float16,
)

print("reached 1")
peft_model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

inputs = [
    "Почему трава зеленая?",
    "Сочини длинный рассказ, обязательно упоминая следующие объекты. Дано: Таня, мяч"
]

for inp in inputs:
    conversation = Conversation()
    conversation.add_user_message(inp)
    prompt = conversation.get_prompt(tokenizer)

    output = generate(peft_model, tokenizer, prompt, generation_config)
    print(inp)
    print(output)
    print()
    print("==============================")
    print()
