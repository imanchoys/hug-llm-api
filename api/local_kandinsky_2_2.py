from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "portrait of a young women, blue eyes, cinematic"
negative_prompt = "low quality, bad quality"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    prior_guidance_scale=1.0,
    height=512,
    width=512
).images[0]

image.save("portrait.png")
