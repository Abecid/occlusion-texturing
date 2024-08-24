import numpy as np
import os

from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import torch
import yaml
import cv2

import config

# Set the device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("Using CUDA")
else:
    print("Using CPU")

# Get depth estimation
def depth_estimation(image):
    depth_estimator = pipeline('depth-estimation')

    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return image

def get_canny_edges(image):
    low_threshold = 100
    high_threshold = 200
    image = np.array(image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return image

# Generate image with controlnet
def generate_style(depth_image, text_condition, num_steps, controlnet="lllyasviel/sd-controlnet-depth", sd_model="runwayml/stable-diffusion-v1-5"):
    controlnet = ControlNetModel.from_pretrained(
        controlnet, torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe.to(device)

    pipe.enable_model_cpu_offload()

    image = pipe(text_condition, depth_image, num_inference_steps=num_steps).images[0]

    return image

def apply_texture(stylized_image):
    pass

def main(config_data):
    depth_model = config_data['controlnet']['depth_model']
    canny_model = config_data['controlnet']['canny_model']
    sd_model = config_data['stable_diffusion']['model']
    num_steps = config_data['stable_diffusion']['num_steps']
    text_condition = config_data['stylization']['text_condition']
    image_path = config_data['stylization']['image_path']
    output_path = config_data['output']['style']
    mesh_path = config_data['asset']['path']
    stylization_type = config_data['stylization']['type']
    controlnet = depth_model if stylization_type == "depth" else canny_model
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]

    image = Image.open(image_path)

    if stylization_type == "depth":
        condition_image = depth_estimation(image)
    elif stylization_type == "canny":
        condition_image = get_canny_edges(image)
    stylized_image = generate_style(condition_image, text_condition, num_steps, controlnet=controlnet, sd_model=sd_model)
    
    depth_filename = f"{stylization_type}_{mesh_name}.png"
    filename = f"stylized_{mesh_name}_{stylization_type}.png"
    style_output_path = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(style_output_path), exist_ok=True)
    stylized_image.save(style_output_path)
    condition_image.save(os.path.join(output_path, depth_filename))

if __name__ == "__main__":
    args = config.get_config()
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    main(config_data)