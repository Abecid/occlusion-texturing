import yaml
import os

import trimesh
import numpy as np
import cv2
import pickle

import config

def apply_styled_colors_to_mesh(mesh, hit_data, _styled_hit_images, image_width=512):
    # Load the styled hit images
    max_hits = len(_styled_hit_images[0])
    for view_idx, view_hit_data in enumerate(hit_data):
        if view_idx >= len(_styled_hit_images):
            break
        styled_hit_images_paths = _styled_hit_images[view_idx][f'view{view_idx+1}']
        print(styled_hit_images_paths)
        styled_hit_images = [cv2.imread(path) for path in styled_hit_images_paths]
        hits_per_ray = view_hit_data        
        
        # print(f"Keys in hits_per_ray: {hits_per_ray.keys()}")
        
        styled_colors = []
        for i in range(max_hits):
            styled_image = styled_hit_images[i]
            styled_image = cv2.cvtColor(styled_image, cv2.COLOR_BGR2RGB) / 255.0
            styled_image = styled_image.reshape(-1, 3)
            styled_colors.append(styled_image)
        
        new_face_colors = mesh.visual.face_colors.copy()
        
        for i in range(max_hits):
            for ray_idx in range(image_width * image_width):
                if i < len(hits_per_ray[ray_idx]):
                    u, v = divmod(ray_idx, image_width)
                    point, normal, color, face_idx = hits_per_ray[ray_idx][i]
                    new_color = styled_colors[i][u * image_width + v]
                    new_face_colors[face_idx] = np.append((new_color * 255).astype(np.uint8), 255)
        
        # Update mesh colors
        mesh.visual.face_colors = new_face_colors
    
    return mesh

def main(config):
    ray_data_path = config_data['texturing']['ray_info']
    mesh_path = config_data['asset']['path']
    style_images = config_data['texturing']['style_images']
    output_path = config_data['output']['textured_mesh']
    
    
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    mesh.visual = mesh.visual.to_color()
    with open(ray_data_path, 'rb') as file:
        list_hits = pickle.load(file)
    new_mesh = apply_styled_colors_to_mesh(mesh, list_hits, style_images)
    
    # save new mesh
    mesh_name = os.path.basename(mesh_path)
    new_mesh_name = f"stylized_{mesh_name}"
    output_path = os.path.join(output_path, new_mesh_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # save the new mesh
    new_mesh.export(output_path)

if __name__ == "__main__":
    args = config.get_config()
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    main(config_data)