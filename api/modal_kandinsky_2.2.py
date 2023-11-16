from pathlib import Path
from typing import Any

from modal import Image, Stub, gpu, method
# from modal import Mount, asgi_app


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
    logger.info(f"Model downloaded: {pipe_path}")


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


stub = Stub(
    "kandinsky-2.2-test-01",
    image=image
)

if stub.is_inside():
    from loguru import logger

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
        negative_prompt: str,
        prior_guidance_scale: float = 1.0,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        num_images_per_prompt: int = 1
    ) -> bytes:
        from loguru import logger
        # TODO: Remove this debug print
        # <class 'diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_combined.KandinskyV22CombinedPipeline'>
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

        # TODO: Remove this debug print
        logger.debug(
            f"Images type is: {type(images)}"
            f" length is: {len(images)}"
        )

        img = images[0]

        # TODO: Remove this debug print
        # 'PIL.Image.Image'
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
