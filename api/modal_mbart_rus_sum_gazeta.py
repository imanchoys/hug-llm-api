from pathlib import Path

from modal import Image, Stub, gpu, method
from modal import Mount, asgi_app

DEFAULT_EXAMPLE = (
    "Высота башни составляет 324 метра (1063 фута), "
    "примерно такая же высота,как у 81-этажного здания, и самое высокое сооружение в Париже. "
    "Его основание квадратно, размером 125 метров (410 футов) с любой стороны. "
    "Во время строительства Эйфелева башня превзошла монумент Вашингтона, "
    "став самым высоким искусственным сооружением в мире, "
    "и этот титул она удерживала в течение 41 года до завершения строительство здания Крайслер в Нью-Йорке в 1930 году. "
    "Это первое сооружение которое достигло высоты 300 метров. "
    "Из-за добавления вещательной антенны на вершине башни в 1957 году "
    "она сейчас выше здания Крайслер на 5,2 метра (17 футов). "
    "За исключением передатчиков, Эйфелева башня является второй самой "
    "высокой отдельно стоящей структурой во Франции после виадука Мийо."
)


def download_mbart_sum_gazeta():
    from transformers import MBartTokenizer, MBartForConditionalGeneration
    # download tokenizer from HF
    _ = MBartTokenizer.from_pretrained("IlyaGusev/mbart_ru_sum_gazeta")
    # download model from HF
    _ = MBartForConditionalGeneration.from_pretrained("IlyaGusev/mbart_ru_sum_gazeta")


image = (
    Image.debian_slim()
    .pip_install(
        "loguru",
        "torch",
        "transformers",
        "sentencepiece"
        # "accelerate"
    )
).run_function(
    # download mbart summarization model
    # for Russian language at image build step
    download_mbart_sum_gazeta
)

with image.run_inside():
    from transformers import (
        MBartTokenizer,
        MBartForConditionalGeneration
    )

stub = Stub(
    "mbart-gazeta-test-01",
    image=image
)


@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
class Model:
    def __enter__(self):
        self.tokenizer = MBartTokenizer.from_pretrained(
            "IlyaGusev/mbart_ru_sum_gazeta"
        )

        self.model = MBartForConditionalGeneration.from_pretrained(
            "IlyaGusev/mbart_ru_sum_gazeta"
        )

    @method()
    def summarize(self, input_text: str) -> str:
        input_ids = self.tokenizer(
            [input_text],
            # максимальная длинна текста на выходе
            max_length=600,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )[0]

        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return summary


@stub.local_entrypoint()
def main(text: str = None):
    from loguru import logger
    if text is None:
        text = DEFAULT_EXAMPLE

    logger.info(f"Input is: {text}")
    summary = Model().summarize.remote(text)
    logger.success(summary)


frontend_path = Path(__file__).parent / "summarizer_frontend"


@stub.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@asgi_app()
def app():
    import fastapi.staticfiles
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from loguru import logger

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )

    logger.debug("Middleware added!")

    @web_app.get("/summary/{text}")
    async def summary(text: str) -> str:
        from fastapi.responses import Response
        summary_res = Model().summarize.remote(text)
        # return Response(
        #     {
        #         "input": text,
        #         "summary": summary_res
        #     },
        #     status_code=200
        # )
        return summary_res
    
    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app
