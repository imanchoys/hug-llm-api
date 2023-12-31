import base64
from io import BytesIO
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from utils import gen_image_name, prep_image_dir

_DEFAULT_MODEL = "kandinsky-community/kandinsky-2-2-decoder"
_DEFAULT_NEG_PROMPT = "low quality, bad quality"
_DEFAULT_IMAGE_DIR = "local_kandinsky_2_2"
_KEY = "kandinsky_2_2"

# this model need GPU and CUDA
if not torch.cuda.is_available():
    raise RuntimeError("Could not run this model without CUDA")

# set up the pipeline for Kandinsky 2.2
pipe: DiffusionPipeline = AutoPipelineForText2Image.from_pretrained(
    _DEFAULT_MODEL,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# init Fast API app
app = FastAPI()

# set up Fast API app
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# save to subdirectory inside /tmp
image_dir: Path = prep_image_dir(_DEFAULT_IMAGE_DIR)


def make_image(
    prompt: str,
    negative_prompt: str,
    prior_guidance_scale: float = 1.0,
    height: int = 512,
    width: int = 512
) -> Image.Image:
    # TODO: Remove this debug print
    # <class 'diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_combined.KandinskyV22CombinedPipeline'>
    print("[DEBUG] Pipe type is:", type(pipe))

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prior_guidance_scale=prior_guidance_scale,
        height=height,
        width=width,
        num_inference_steps=25
    ).images

    # TODO: Remove this debug print
    print("[DEBUG] Images object has type:", type(pipe), "len is:", len(images))
    img = images[0]

    # TODO: Remove this debug print
    # 'PIL.Image.Image'
    print("[DEBUG] Single image has type:", type(img))

    return img


@app.get("/")
def generate(
    prompt: str,
    height: int,
    width: int,
    guidance_scale: float = 1.0,
    neg_prompt: str = _DEFAULT_NEG_PROMPT,
    save_to_fs: bool = True
):
    # generate image
    image: Image.Image = make_image(
        prompt,
        neg_prompt,
        prior_guidance_scale=guidance_scale,
        height=height,
        width=width
    )

    # save to fs
    if save_to_fs:
        # generate filename for the image
        # (Kandinsky default format is .png)
        # TODO: check if other formats supported
        filename = gen_image_name(
            key=_KEY,
            file_extension="png"
        )
        # save the image (Kandinsky default format is .png)
        # TODO: check if other formats supported
        image.save(image_dir / filename)

    # save to buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())

    # Fast API image response
    return Response(content=img_str, media_type="image/png")
