import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import cv2
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import yaml

import config


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
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape) # (H, W, 3)

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
    if isinstance(image_resolution, tuple):
        assert len(image_resolution) == 2
    else:
        image_resolution = (image_resolution, image_resolution)
    image_width, image_height = image_resolution

    # generate an array of screen coordinates for the rays
    # (rays are placed at locations [i, j] in the image)
    rays_screen_coords = np.mgrid[0:image_height, 0:image_width].reshape(
        2, image_height * image_width).T  # [h, w, 2]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    grid = rays_screen_coords.reshape(image_height, image_width, 2)
    
    i, j = grid[..., 1], grid[..., 0]
    directions = np.stack([(i-cx)/fx, -(j-cy)/fy, -np.ones_like(i)], -1) # (H, W, 3)

    rays_origins, ray_directions = get_rays(directions, c2w)
    rays_origins = rays_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    
    return rays_screen_coords, rays_origins, ray_directions


def ray_cast_mesh(mesh, rays_origins, ray_directions):
    intersector = RayMeshIntersector(mesh)
    index_triangles, index_ray, point_cloud = intersector.intersects_id(
        ray_origins=rays_origins,
        ray_directions=ray_directions,
        multiple_hits=True,
        return_locations=True)
    return index_triangles, index_ray, point_cloud

def save_plane_images(model_path, views, camera_angle_x, max_hits, output_path, image_height, image_width):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_path = os.path.join(output_path, model_name)
    os.makedirs(output_path, exist_ok=True)

    try:
        mesh = trimesh.load(model_path,  force='mesh', process=False)

        mesh.visual = mesh.visual.to_color()
        
        camera_angle_x = float(camera_angle_x)
        for idx, view in tqdm(enumerate(views), total=len(views)):
            azimuth, elevation, radius = view['azimuth'], view['elevation'], view['radius']
            c2w = generate_c2w_matrix(azimuth, elevation, radius)
            focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

            cx = image_width / 2.0
            cy = image_height / 2.0

            intrinsics = np.array([[focal_length, 0, cx],
                                   [0, focal_length, cy],
                                   [0, 0, 1]])

            rays_origins, ray_directions = generate_rays((image_width, image_height), intrinsics, c2w)

            ray_indexes, points, normals, colors = ray_cast_mesh(mesh, rays_origins, ray_directions)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            colors = colors[:, :3] / 255.0

            # collect points and normals for each ray
            ray_points = defaultdict(list)
            ray_normals = defaultdict(list)
            ray_colors = defaultdict(list)
            for ray_index, point, normal, color in zip(ray_indexes, points, normals, colors):
                ray_points[ray_index].append(point)
                ray_normals[ray_index].append(normal)
                ray_colors[ray_index].append(color)

            # ray to image
            GenDepths = np.ones((max_hits, 1 + 3 + 3, image_height, image_width), dtype=np.float32)
            GenDepths[:, :4, :, :] = 0 # set depth and normal to zero while color is by default 1 (white)

            for i in range(max_hits):
                for ray_index, ray_point in ray_points.items():
                    if i < len(ray_point):
                        u = ray_index // image_width
                        v = ray_index % image_width
                        GenDepths[i, 0, u, v] = np.linalg.norm(ray_point[i] - c2w[:, 3])
                        GenDepths[i, 1:4, u, v] = ray_normals[ray_index][i]
                        GenDepths[i, 4:7, u, v] = ray_colors[ray_index][i]

            # Save Images
            save_dir = os.path.join(output_path, f"{idx}_azim_{azimuth}_elevation_{elevation}_radius_{radius}")
            os.makedirs(save_dir, exist_ok=True)

            for i in range(max_hits):
                color_image = (GenDepths[i, 4:7] * 255).astype(np.uint8)
                color_image = np.transpose(color_image, (1, 2, 0)) # (h, w, c)
                image_path = os.path.join(save_dir, f"hit_{i}.png")
                cv2.imwrite(image_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

        print("saved")
    except Exception as e:
        print(e)
        return
                

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
