import torch

from diffusers import DiffusionPipeline, AutoencoderKL


def inference(checkpoint):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True
    )
    # pipe.load_lora_weights("felixdae/lora-trained-xl-colab")
    pipe.load_lora_weights(checkpoint)

    _ = pipe.to("cuda")

    prompt = "a photo of sks dog in a bucket"
    # prompt = "a photo of sks dog on the grass"

    image = pipe(prompt=prompt, num_inference_steps=50).images[0]
    return image
