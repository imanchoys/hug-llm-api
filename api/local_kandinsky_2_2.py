import os
from typing import Any
import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path
from utils import gen_image_name

DEFAULT_MODEL = "kandinsky-community/kandinsky-2-2-decoder"
DEFAULT_NEG_PROMPT = "low quality, bad quality"
DEFAULT_IMG_FOLDER = "local_kandinsky_2_2"

# this model need GPU and CUDA
if not torch.cuda.is_available():
    raise RuntimeError("Could not run this model without CUDA")

# set up the pipeline for Kandinsky 2.2
pipe = AutoPipelineForText2Image.from_pretrained(
    DEFAULT_MODEL,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")


def generate_image(
    prompt: str,
    negative_prompt: str,
    height: int = 512,
    width: int = 512
) -> Any:
    # TODO: Remove this debug print
    print("[DEBUG] Pipe types is:", type(pipe))

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prior_guidance_scale=1.0,
        height=height,
        width=width
    ).images[0]

    # TODO: Remove this debug print
    print("[DEBUG] Pipe types is:", type(pipe))

    return image


if __name__ == "__main__":
    prompt = "portrait of a young women, red eyes, cinematic"
    # by default folder would be created inside UNIX `/tmp` dir
    path_prefix = f"/tmp/{DEFAULT_IMG_FOLDER}"

    # create specified directory if not present
    dir = Path(path_prefix)
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    # show path where generated images would be saved
    print("Images would be saved to:", dir)

    image = generate_image(
        prompt,
        DEFAULT_NEG_PROMPT
    )

    # generate filename for the image
    # (Kandinsky default format is .png)
    filename = gen_image_name(
        key="local_kandinsky_2_2",
        file_extension="png"
    )

    # save the image (Kandinsky default format is .png)
    img_path = dir / filename
    image.save(img_path)
