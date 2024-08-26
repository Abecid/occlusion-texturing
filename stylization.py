import numpy as np
import os

from transformers import pipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers import UniPCMultistepScheduler, DDPMScheduler
from PIL import Image
import torch
import yaml
import cv2

import config
from ip_adapter import IPAdapterXL, IPAdapter
# from texturing import StyleTexturingPipeline as stp

# Set the device variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("Using CUDA")
else:
    print("Using CPU")
    
SEED = 42

def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    # Threshold parameters from Instant Style: https://github.com/InstantStyle/InstantStyle/blob/f69273512cdf4efa09737f8906d61d981791396d/infer_style_controlnet.py#L39
    low_threshold = 50
    high_threshold = 200
    image = np.array(image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return image

def apply_styled_colors_to_mesh(mesh, hit_data, styled_hit_images):
    # Load the styled hit images
    max_hits = len(styled_hit_images)
    for view_idx, view_hit_data in enumerate(hit_data):
        hits_per_ray = view_hit_data['hits_per_ray']
        image_height = view_hit_data['image_height']
        image_width = view_hit_data['image_width']
        
        styled_colors = []
        for i in range(max_hits):
            styled_image = cv2.imread(styled_hit_images[i])
            styled_image = cv2.cvtColor(styled_image, cv2.COLOR_BGR2RGB) / 255.0
            styled_image = styled_image.reshape(-1, 3)
            styled_colors.append(styled_image)
        
        new_face_colors = mesh.visual.face_colors
        
        for ray_idx, hits in hits_per_ray.items():
            for i in range(max_hits):
                if i < len(hits):
                    u, v = divmod(ray_idx, image_width)
                    point, normal, original_color, face_idx = hits[i]
                    new_color = styled_colors[i][u * image_width + v]
                    new_face_colors[face_idx] = (new_color * 255).astype(np.uint8)
        
        # Update mesh colors
        mesh.visual.face_colors = new_face_colors
    
    return mesh

# Generate image with controlnet
def generate_style(
    condition_image, 
    text_condition,
    num_steps,
    controlnet="lllyasviel/sd-controlnet-depth",
    sd_model="runwayml/stable-diffusion-v1-5",
    ip_adapter_ckpt_path="models/ip-adapter_sdxl.bin",
    image_encoder="models/image_encoder",
    style_image=None,
    controlnet_conditioning_scale=0.8
):
    controlnet = ControlNetModel.from_pretrained(
        controlnet, torch_dtype=torch.float16
    )

    if 'xl' in sd_model:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            sd_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config) # SD-1.5
        # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        
    pipe.enable_vae_tiling()
    
    pipe.to(device)

    # pipe.enable_model_cpu_offload()    
    
    if style_image is None:
        print(f"Generating image with text condition: {text_condition}")
        image = pipe(text_condition, condition_image, num_inference_steps=num_steps).images[0]
        return image

    style_image.resize((512, 512))
    
    # load ip-adapter
    original_target_blocks = ["block"] # for original IP-Adapter
    style_target_blocks = ["up_blocks.0.attentions.1"]
    style_layout_target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
    
    if 'xl' in ip_adapter_ckpt_path:
        ip_model = IPAdapterXL(pipe, image_encoder, ip_adapter_ckpt_path, device, target_blocks=style_target_blocks)
    else:
        ip_model = IPAdapter(pipe, image_encoder, ip_adapter_ckpt_path, device, target_blocks=original_target_blocks)
    
    images = ip_model.generate(
        pil_image=style_image,
        prompt=text_condition,
        negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        scale=1.0,
        guidance_scale=5,
        num_samples=1,
        num_inference_steps=30, 
        seed=SEED,
        image=condition_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )
    
    image = images[0]

    return image

def main(config_data):
    depth_model = config_data['controlnet']['depth_model']
    canny_model = config_data['controlnet']['canny_model']
    sd_model = config_data['stable_diffusion']['model']
    num_steps = config_data['stable_diffusion']['num_steps']
    text_condition = config_data['stylization']['text_condition']
    base_image_path = config_data['stylization']['base_image_path']
    style_image_path = config_data['stylization']['style_image_path']
    output_path = config_data['output']['style']
    mesh_path = config_data['asset']['path']
    stylization_type = config_data['stylization']['type']
    ip_adapter_ckpt_path = config_data['ip_adapter']['ckpt_path']
    image_encoder = config_data['ip_adapter']['image_encoder']
    controlnet_conditioning_scale = config_data['controlnet']['conditioning_scale']
    
    controlnet = depth_model if stylization_type == "depth" else canny_model
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]

    image = Image.open(base_image_path)
    style_image = Image.open(style_image_path)

    if stylization_type == "depth":
        condition_image = depth_estimation(image)
    elif stylization_type == "canny":
        condition_image = get_canny_edges(image)
    stylized_image = generate_style(
        condition_image,
        text_condition,
        num_steps,
        controlnet=controlnet,
        sd_model=sd_model,
        ip_adapter_ckpt_path=ip_adapter_ckpt_path,
        image_encoder=image_encoder,
        style_image=style_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale
    )
    
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