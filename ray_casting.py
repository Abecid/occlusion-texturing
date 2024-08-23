import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import cv2
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import yaml

import config


def normalize_mesh(mesh):
    # Get the bounding box extents of the mesh (width, height, depth)
    extents = mesh.bounding_box.extents
    max_extent = np.max(extents)

    # Scale the mesh so that its maximum extent is 1
    scale_factor = 1.0 / max_extent
    mesh.apply_scale(scale_factor)

    return mesh, scale_factor

def generate_c2w_matrix(azimuth, elevation, radius):
    # Convert degrees to radians
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    # Calculate camera position in spherical coordinates
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)

    # Camera position (translation)
    translation = np.array([x, y, z])

    # Define the camera orientation (rotation)
    forward = -translation / np.linalg.norm(translation)  # Camera looks towards the origin
    right = np.cross(np.array([0, 1, 0]), forward)        # Compute the right vector
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    # Construct the c2w matrix
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = translation

    return c2w

def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
    return rays_o, rays_d

class RaycastingImaging:
    def __init__(self):
        self.rays_screen_coords, self.rays_origins, self.rays_directions = None, None, None

    def __del__(self):
        del self.rays_screen_coords
        del self.rays_origins
        del self.rays_directions

    def prepare(self, image_height, image_width, intrinsics=None, c2w=None):
        # scanning radius is determined from the mesh extent
        self.rays_screen_coords, self.rays_origins, self.rays_directions = generate_rays((image_height, image_width), intrinsics, c2w)

    def get_image(self, mesh):  #, features):
        # get a point cloud with corresponding indexes
        mesh_face_indexes, ray_indexes, points = ray_cast_mesh(mesh, self.rays_origins, self.rays_directions)

        # extract normals
        normals = mesh.face_normals[mesh_face_indexes]
        colors = mesh.visual.face_colors[mesh_face_indexes]

        mesh_face_indexes = np.unique(mesh_face_indexes)
        mesh_vertex_indexes = np.unique(mesh.faces[mesh_face_indexes])
        direction = self.rays_directions[ray_indexes][0]
        return ray_indexes, points, normals, colors, direction, mesh_vertex_indexes, mesh_face_indexes


def generate_rays(image_resolution, intrinsics, c2w):
    h, w = image_resolution
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    directions = np.stack([(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], axis=-1)
    
    rays_o, rays_d = get_rays(directions, c2w)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def ray_cast_mesh(mesh, rays_o, rays_d):
    intersector = RayMeshIntersector(mesh)
    index_triangles, index_ray, point_cloud = intersector.intersects_id(
        ray_origins=rays_o,
        ray_directions=rays_d,
        multiple_hits=True,
        return_locations=True
    )
    return index_triangles, index_ray, point_cloud

def save_plane_images(model_path, views, camera_angle_x, max_hits, output_path, image_height, image_width):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_path = os.path.join(output_path, model_name)
    os.makedirs(output_path, exist_ok=True)

    mesh = trimesh.load(model_path, force='mesh', process=False)
    mesh.visual = mesh.visual.to_color()
    mesh, scale_factor = normalize_mesh(mesh)
    print(f"Scaled mesh by {scale_factor}")
    
    camera_angle_x = float(camera_angle_x)
    for view_index, view in tqdm(enumerate(views), total=len(views)):
        azimuth, elevation, radius = view['azimuth'], view['elevation'], view['radius']
        c2w = generate_c2w_matrix(azimuth, elevation, radius)
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0

        intrinsics = np.array([[focal_length, 0, cx],
                                [0, focal_length, cy],
                                [0, 0, 1]])

        rays_o, rays_d = generate_rays((image_height, image_width), intrinsics, c2w)

        index_triangles, index_ray, points = ray_cast_mesh(mesh, rays_o, rays_d)
        normals = mesh.face_normals[index_triangles]
        colors = mesh.visual.face_colors[index_triangles][:, :3] / 255.0

        # ray to image
        GenDepths = np.ones((max_hits, 1 + 3 + 3, image_height, image_width), dtype=np.float32)
        GenDepths[:, :4, :, :] = 0 # set depth and normal to zero while color is by default 1 (white)

        hits_per_ray = defaultdict(list)
        for idx, ray_idx in enumerate(index_ray):
            hits_per_ray[ray_idx].append((points[idx], normals[idx], colors[idx]))

        # hits_per_ray[ray_idx].sort(key=lambda hit: np.linalg.norm(hit[0] - c2w[:3, 3]))  # Sort by depth

        # Populate the hit images
        for i in range(max_hits):
            for ray_idx in range(image_height * image_width):
                if i < len(hits_per_ray[ray_idx]):
                    u, v = divmod(ray_idx, image_width)
                    point, normal, color = hits_per_ray[ray_idx][i]
                    depth = np.linalg.norm(point - c2w[:3, 3])

                    GenDepths[i, 0, u, v] = depth
                    GenDepths[i, 1:4, u, v] = normal
                    GenDepths[i, 4:7, u, v] = color

        # Save Images
        save_dir = os.path.join(output_path, f"{view_index}_{azimuth}_{elevation}_{radius}")
        os.makedirs(save_dir, exist_ok=True)

        for i in range(max_hits):
            color_image = (GenDepths[i, 4:7] * 255).astype(np.uint8)
            color_image = np.transpose(color_image, (1, 2, 0)) # (h, w, c)
            image_path = os.path.join(save_dir, f"hit_{i}.png")
            cv2.imwrite(image_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

    print("saved")
                

if __name__ == "__main__":
    args = config.get_config()
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)

    model_path = config_data['asset']["path"]
    views = config_data['rendering']['views']
    camera_angle_x = config_data['camera']['angle_x']
    max_hits = config_data['rendering']['max_hits']
    output_path = config_data['output']['rc']
    image_height = config_data['rendering']['height']
    image_width = config_data['rendering']['width']

    save_plane_images(model_path, views, camera_angle_x, max_hits, output_path, image_height, image_width)
