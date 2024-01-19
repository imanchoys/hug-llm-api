from pathlib import Path

from modal import Image, Stub, gpu, method
from modal import Mount, asgi_app

# TODO: Improve set up for Kandinsky 2.2 model
# example: https://colab.research.google.com/drive/1MfN9dfmejT8NjXhR353NeP5RzbruHgo7?usp=sharing#scrollTo=lx6QEDLd7ZDz


def download_via_AutoPipeline():
    import torch
    from diffusers import (
        AutoPipelineForText2Image,
        DiffusionPipeline,
        KandinskyV22CombinedPipeline
    )
    from loguru import logger
    pipe: KandinskyV22CombinedPipeline | DiffusionPipeline = AutoPipelineForText2Image.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder",
        torch_dtype=torch.float16
    )
    pipe_path = pipe.download("kandinsky-community/kandinsky-2-2-decoder")
    logger.info(f"Model downloaded successfully: {pipe_path}")


image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1"
    )
    .pip_install(
        "loguru",
        "torch",
        "diffusers~=0.23",
        "transformers",
        "accelerate~=0.24"
    )
    .run_function(download_via_AutoPipeline)
)

# TODO: explore how to manage imports properly
# with image.run_inside():
#     from loguru import logger

stub = Stub(
    "kandinsky-2.2-test-01",
    image=image
)


@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
class Model:
    def __enter__(self):
        from loguru import logger
        import torch
        from diffusers import (
            AutoPipelineForText2Image,
            DiffusionPipeline,
            KandinskyV22CombinedPipeline
        )

        # this model need GPU and CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("Could not run this model without CUDA")

        options = dict(
            torch_dtype=torch.float16
        )

        # Load base model
        self.kandinsky_pipe: DiffusionPipeline | KandinskyV22CombinedPipeline = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            **options
        )

        self.kandinsky_pipe.to("cuda")
        logger.debug("Pipeline initialized")

    @method()
    def inference(
        self,
        prompt: str,
        negative_prompt: str = "low quality, bad quality",
        prior_guidance_scale: float = 1.0,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        num_images_per_prompt: int = 1
    ) -> bytes:
        from loguru import logger
        logger.debug(f"Pipe type is: {type(self.kandinsky_pipe)}")

        images = self.kandinsky_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prior_guidance_scale=prior_guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt
        ).images

        logger.debug(
            f"Images type is: {type(images)}"
            f" length is: {len(images)}"
        )

        img = images[0]
        logger.debug(f"Single image has type: {type(img)}")

        import io
        byte_stream = io.BytesIO()
        img.save(byte_stream, format="PNG")
        return byte_stream.getvalue()


@stub.local_entrypoint()
def main(prompt: str):
    from loguru import logger
    image_bytes = Model().inference.remote(
        prompt,
        "low quality, bad quality"
    )

    dir = Path("/tmp/kandinsky-2.2-img-dir")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    logger.info(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)


frontend_path = Path(__file__).parent / "frontend"


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

    @web_app.get("/infer/{prompt}")
    async def infer(prompt: str):
        from fastapi.responses import Response

        image_bytes = Model().inference.remote(prompt)

        return Response(image_bytes, media_type="image/png")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app
