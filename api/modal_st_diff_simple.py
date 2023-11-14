# why? we have python 3.10+
from __future__ import annotations

import io
import time
from pathlib import Path

# import modal-specific things
from modal import Image, Stub, method

# give app a name
stub = Stub("stable-diffusion-cli")

glob_model_id = "runwayml/stable-diffusion-v1-5"
glob_cache_path = "/vol/cache"


def download_models():
    import diffusers
    import torch

    # Download scheduler configuration. Experiment with different schedulers
    # to identify one that works best for your use-case.
    scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
        glob_model_id,
        subfolder="scheduler",
        cache_dir=glob_cache_path,
    )
    scheduler.save_pretrained(glob_cache_path, safe_serialization=True)

    # Downloads all other models.
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        glob_model_id,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir=glob_cache_path,
    )
    pipe.save_pretrained(glob_cache_path, safe_serialization=True)


# would run once, at image creation
# we are "baking" the dependencies into the image
image = (
    Image.debian_slim(python_version="3.10")  # local PYTHON version: 3.11.5
    .pip_install(
        "accelerate",               # local 'accelerate':       0.24.1
        "diffusers[torch]>=0.15.1", # local 'diffusers':       0.23.0
        "ftfy",                     # local 'ftfy':             6.1.1
        "torchvision",              # local 'torchvision':      0.16.0
        "transformers~=4.25.1",     # local 'transformers':     4.35.0
        "triton",                   # local 'triton':           2.1.0
        "safetensors",              # local 'safetensors':      0.4.0
    )
    .pip_install(
        "torch==2.0.1+cu117",
        find_links="https://download.pytorch.org/whl/torch_stable.html",
    )
    .pip_install("xformers", pre=True)
    # this would download the model at image build
    .run_function(download_models)
)

# set stub's image to the one we've initialized
stub.image = image


@stub.cls(gpu="A10G")
class StableDiffusion:
    def __enter__(self):
        import diffusers
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True

        scheduler = diffusers.DPMSolverMultistepScheduler.from_pretrained(
            glob_cache_path,
            subfolder="scheduler",
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            denoise_final=True,  # important if steps are <= 10
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            glob_cache_path,
            scheduler=scheduler,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.pipe.enable_xformers_memory_efficient_attention()

    @method()
    def run_inference(
        self, prompt: str, steps: int = 20, batch_size: int = 4
    ) -> list[bytes]:
        import torch

        with torch.inference_mode():
            with torch.autocast("cuda"):
                images = self.pipe(
                    [prompt] * batch_size,
                    num_inference_steps=steps,
                    guidance_scale=7.0,
                ).images

        # Convert to PNG bytes
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        return image_output


@stub.local_entrypoint()
def entrypoint(
    prompt: str,
    samples: int = 5,
    steps: int = 10,
    batch_size: int = 1,
    folder_name: str = "modal_st_diff_simple"
):
    """
    This is the command we'll use to generate images. It takes a `prompt`,
    `samples` (the number of images you want to generate), `steps` which
    configures the number of inference steps the model will make, and `batch_size`
    which determines how many images to generate for a given prompt.
    """

    # by default folder would be created inside UNIX `/tmp` dir
    path_prefix = f"/tmp/{folder_name}"

    # show generation parameters
    print(
        f"prompt => '{prompt}', steps => {steps}, "
        f"samples => {samples}, batch_size => {batch_size}"
    )

    # create specified directory if not present
    dir = Path(path_prefix)
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    # show path where generated images would be saved
    print("Images would be saved to:", dir)

    sd = StableDiffusion()
    for i in range(samples):
        t0 = time.time()
        images = sd.run_inference.remote(prompt, steps, batch_size)
        total_time = time.time() - t0
        print(
            f"Sample {i} took {total_time:.3f}s ({(total_time)/len(images):.3f}s / image)."
        )
        for j, image_bytes in enumerate(images):
            output_path = dir / f"output_{j}_{i}.png"
            print(f"Saving it to {output_path}")
            with open(output_path, "wb") as f:
                f.write(image_bytes)
