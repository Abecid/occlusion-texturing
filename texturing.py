import numpy as np
import os

from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import torch
import yaml

import config

image_path = ""

# Get depth estimation
def depth_estimation(image_path):
    depth_estimator = pipeline('depth-estimation')

    image = Image.open(image_path)

    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return image

# Generate image with controlnet
def generate_texture(depth_image, text_condition, num_steps, depth_model="lllyasviel/sd-controlnet-depth", sd_model="runwayml/stable-diffusion-v1-5"):
    controlnet = ControlNetModel.from_pretrained(
        depth_model, torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()

    image = pipe(text_condition, depth_image, num_inference_steps=num_steps).images[0]

    return image

def apply_texture(texture_image):
    pass

def main(config_data):
    depth_model = config_data['depth_model']
    sd_model = config_data['stable_diffusion']['model']
    num_steps = config_data['stable_diffusion']['num_steps']
    text_condition = config_data['text_condition']
    output_path = config_data['output']['texture']
    mesh_path = config_data['asset']['path']

    depth_image = depth_estimation(image_path)
    texture_image = generate_texture(depth_image, text_condition, num_steps)
    
    filename = "texture.png"
    output_path = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    texture_image.save(output_path)

if __name__ == "__main__":
    args = config.get_config()
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    main(config_data)