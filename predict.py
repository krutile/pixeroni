# predict.py
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import cog
import os

class Predictor(cog.Predictor):
    def setup(self):
        # Load ControlNet models
        controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        controlnet_tile = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-tile", torch_dtype=torch.float16)

        # Load the base model (EpicRealism or SD1.5)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "emilianJR/epicrealism-naturalsinRC1vae",
            controlnet=[controlnet_canny, controlnet_tile],
            torch_dtype=torch.float16
        ).to("cuda")

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

    @cog.input("prompt", type=str, default="masterpiece, best quality, highres, beautiful trees, lush greenery")
    @cog.input("negative_prompt", type=str, default="(worst quality, low quality, normal quality:2) JuggernautNegative-neg")
    @cog.input("steps", type=int, default=30)
    @cog.input("cfg_scale", type=float, default=2.0)
    @cog.input("denoising_strength", type=float, default=0.35)
    @cog.input("seed", type=int, default=42)
    @cog.input("image", type=Image, help="Optional input image for ControlNet", default=None)
    def predict(self, prompt, negative_prompt, steps, cfg_scale, denoising_strength, seed, image):
        generator = torch.manual_seed(seed)

        # Use a blank canny-style image if none provided
        if image is None:
            image = Image.new("RGB", (768, 512), (128, 128, 128))

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            image=image,
            generator=generator
        )

        output_image = output.images[0]
        output_path = "/tmp/output.png"
        output_image.save(output_path)
        return output_path
