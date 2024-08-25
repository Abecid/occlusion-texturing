import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import cv2
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import yaml
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import config


def visualize_rays_and_intersections(mesh, rays_o, rays_d, points, num_vis=100, ray_length = 1.5, save_dir='./'):
    # Plot the mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], color='gray', s=0.5, alpha=0.5)

    # Plot the rays
    for ray_o, ray_d in zip(rays_o[:num_vis], rays_d[:num_vis]):
        ax.plot(
            [ray_o[0], ray_o[0] + ray_d[0] * ray_length], 
            [ray_o[1], ray_o[1] + ray_d[1] * ray_length], 
            [ray_o[2], ray_o[2] + ray_d[2] * ray_length], 
            color='blue', alpha=0.5
        )

    # Plot the intersection points
    if points is not None and len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=5)

    # Set labels and view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=30, azim=45)

    ax.grid(False)

    plt.show()
    filename = "ray_visualization.png"
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path)


def normalize_mesh(mesh):
    # Get the bounding box extents of the mesh (width, height, depth)
    extents = mesh.bounding_box.extents
    max_extent = np.max(extents)

    # Scale the mesh so that its maximum extent is 1
    scale_factor = 1.0 / max_extent
    mesh.apply_scale(scale_factor)

    return mesh, scale_factor

def normalize_to_box(pcd):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    pcd: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(pcd.shape) == 2:
        axis = 0
        P = pcd.shape[0]
        D = pcd.shape[1]
    elif len(pcd.shape) == 3:
        axis = 1
        P = pcd.shape[1]
        D = pcd.shape[2]
    else:
        raise ValueError()
    
    if isinstance(pcd, np.ndarray):
        maxP = np.amax(pcd, axis=axis, keepdims=True)
        minP = np.amin(pcd, axis=axis, keepdims=True)
        centroid = (maxP+minP)/2
        pcd = pcd - centroid
        furthest_distance = np.amax(np.abs(pcd), axis=(axis, -1), keepdims=True)
        scale = 1 / furthest_distance
        pcd = pcd * scale
    elif isinstance(pcd, torch.Tensor):
        maxP = torch.max(pcd, dim=axis, keepdim=True)[0]
        minP = torch.min(pcd, dim=axis, keepdim=True)[0]
        centroid = (maxP+minP)/2
        pcd = pcd - centroid
        in_shape = list(pcd.shape[:axis])+[P*D]
        furthest_distance = torch.max(torch.abs(pcd).view(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        scale = 1 / furthest_distance
        pcd = pcd * scale
    else:
        raise ValueError()

    return pcd, centroid, scale

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
    forward = translation / np.linalg.norm(translation)  # Camera looks towards the origin
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
    
    vertices = mesh.vertices
    vertices, centroid, scale = normalize_to_box(vertices)
    centroid = np.mean(vertices, axis=0)
    vertices -= centroid
    mesh.vertices = vertices
    
    # mesh, scale_factor = normalize_mesh(mesh)
    print(f"Centroid: {centroid}, Scaling: {scale}")
    print(f"Mesh extents: {mesh.bounding_box.extents}")
    
    camera_angle_x = float(camera_angle_x)
    
    focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

    cx = image_width / 2.0
    cy = image_height / 2.0

    intrinsics = np.array([[focal_length, 0, cx],
                            [0, focal_length, cy],
                            [0, 0, 1]])
    
    hit_data = []
    for view_index, view in tqdm(enumerate(views), total=len(views)):
        azimuth, elevation, radius = view['azimuth'], view['elevation'], view['radius']
        c2w = generate_c2w_matrix(azimuth, elevation, radius)
        camera_position = c2w[:3, 3]
        print(f"Camera Position for View {view_index}: {camera_position}")
        
        rays_o, rays_d = generate_rays((image_height, image_width), intrinsics, c2w)

        index_triangles, index_ray, points = ray_cast_mesh(mesh, rays_o, rays_d)
        normals = mesh.face_normals[index_triangles]
        colors = mesh.visual.face_colors[index_triangles][:, :3] / 255.0

        # Save Images
        save_dir = os.path.join(output_path, f"{view_index}_{azimuth}_{elevation}_{radius}")
        os.makedirs(save_dir, exist_ok=True)

        # Call the visualization function after ray_cast_mesh
        visualize_rays_and_intersections(mesh, rays_o, rays_d, points, save_dir=save_dir)
        print(f"Number of rays: {rays_o.shape[0]}")
        print(f"Number of intersections: {len(points)}")

        # ray to image
        GenDepths = np.ones((max_hits, 1 + 3 + 3 + 1, image_height, image_width), dtype=np.float32)
        GenDepths[:, :4, :, :] = 0 # set depth and normal to zero while color is by default 1 (white)

        hits_per_ray = defaultdict(list)
        for idx, ray_idx in enumerate(index_ray):
            hits_per_ray[ray_idx].append((points[idx], normals[idx], colors[idx], index_triangles[idx]))

        # hits_per_ray[ray_idx].sort(key=lambda hit: np.linalg.norm(hit[0] - c2w[:3, 3]))  # Sort by depth
        
        # Store hit data for later use
        hit_data.append({
            'hits_per_ray': hits_per_ray,
            'c2w': c2w,
            'image_height': image_height,
            'image_width': image_width,
            'face_indices': index_triangles,
            'ray_indices': index_ray,
        })

        # Populate the hit images
        for i in range(max_hits):
            for ray_idx in range(image_height * image_width):
                if i < len(hits_per_ray[ray_idx]):
                    u, v = divmod(ray_idx, image_width)
                    point, normal, color, face_idx = hits_per_ray[ray_idx][i]
                    depth = np.linalg.norm(point - c2w[:3, 3])

                    GenDepths[i, 0, u, v] = depth
                    GenDepths[i, 1:4, u, v] = normal
                    GenDepths[i, 4:7, u, v] = color
                    GenDepths[i, 7, u, v] = face_idx

        for i in range(max_hits):
            color_image = (GenDepths[i, 4:7] * 255).astype(np.uint8)
            color_image = np.transpose(color_image, (1, 2, 0)) # (h, w, c)
            image_path = os.path.join(save_dir, f"hit_{i}.png")
            cv2.imwrite(image_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

    print("saved")
    return hit_data
                

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
