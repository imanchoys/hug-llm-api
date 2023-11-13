import base64
import torch
from utils import get_image_name
from torch import autocast
from diffusers import StableDiffusionPipeline
# the line below is for the sake if type annotation of pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline as StDiffPipelineT
)
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
# from typing import Any
# TODO: use environment variable instead of `_secrets.py`
from _secrets import TOKEN_HUGGINGFACE

_DEVICE = "cuda"
_MODEL = "CompVis/stable-diffusion-v1-4"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


def get_pipeline(hf_access_token: str, model_id: str) -> StDiffPipelineT | None:
    """
    Set parameters and get the StDiff pipeline from diffusers
    """
    pl = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=hf_access_token
    )

    print(f"Pipeline for '{model_id}'", type(pl))
    return pl


pipe = get_pipeline(
    TOKEN_HUGGINGFACE,
    _MODEL
)

assert pipe is not None, \
    f"Pipeline object not set or failed to initialize"

pipe.to(_DEVICE)


@app.get("/")
def generate(prompt: str):
    with autocast(_DEVICE):
        image = pipe(prompt, guidance_scale=8.5).images[0]
        image.save(get_image_name(key="user0", file_extension="png"))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())

        return Response(content=img_str, media_type="image/png")
