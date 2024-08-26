import os

import torch
from diffusers.utils import numpy_to_pil
from torchvision.transforms import Resize, InterpolationMode

from renderer.project import UVProjection as UVP

class StyleTexturingPipeline():
    def __init__(
        self, 
        mesh_path,
        camera_angles,
        texture_size,
        render_size,
        camera_centers=None,
        mesh_transform={"scale": 1},
        mesh_autouv=True,
        texture_rgb_size=512,
        render_rgb_size=512
    ):
        self.mesh_path = mesh_path
        self.camera_angles = camera_angles
        self.texture_size = texture_size
        self.render_size = render_size
        self.camera_centers = camera_centers
        self.uvp = None
        self.uvp_rgb = None
        self._execution_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mesh_transform = mesh_transform
        self.mesh_autouv = mesh_autouv
        self.texture_rgb_size = texture_rgb_size
        self.render_rgb_size = render_rgb_size
    
        # Set up pytorch3D for projection between screen space and UV space
        # uvp is for latent and uvp_rgb for rgb color
        self.uvp = UVP(texture_size=self.texture_size, render_size=self.render_size, sampling_mode="nearest", channels=4, device=self._execution_device)
        if self.mesh_path.lower().endswith(".obj"):
            self.uvp.load_mesh(self.mesh_path, scale_factor=self.mesh_transform["scale"] or 1, autouv=self.mesh_autouv)
        elif self.mesh_path.lower().endswith(".glb"):
            self.uvp.load_glb_mesh(self.mesh_path, scale_factor=self.mesh_transform["scale"] or 1, autouv=self.mesh_autouv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
        self.uvp.set_cameras_and_render_settings(self.camera_poses, centers=self.camera_centers, camera_distance=4.0)


        self.uvp_rgb = UVP(texture_size=self.texture_rgb_size, render_size=self.render_rgb_size, sampling_mode="nearest", channels=3, device=self._execution_device)
        self.uvp_rgb.mesh = self.uvp.mesh.clone()
        self.uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=self.camera_centers, camera_distance=4.0)
        _,_,_,cos_maps,_, _ = self.uvp_rgb.render_geometry()
        self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)

        # Render mesh to get the rendered images
        mesh_dir = f"{'/'.join(self.mesh_path.split('/')[:-1])}/init_mesh"
        os.makedirs(mesh_dir, exist_ok=True)

        _,_,_,cos_maps,_, _ = self.uvp.render_geometry()
        rendered_images = self.uvp.render_mesh()
        # rendered_images = self.uvp.render_textured_views()
        rendered = rendered_images[..., :3].cpu().numpy()
        for i in range(len(rendered)):
            numpy_to_pil(rendered[i])[0].save(f"{mesh_dir}/init_mesh_{i}.jpg")
        rendered = np.concatenate([img for img in rendered], axis=1)
        numpy_to_pil(rendered)[0].save(f"{mesh_dir}/init_mesh.jpg")
        print(f"Initial mesh rendered images saved at {mesh_dir}")
        

        # Save some VRAM
        del _, cos_maps
        self.uvp.to("cpu")
        self.uvp_rgb.to("cpu")
        pass
    
    # Decode each view and bake them into a rgb texture
    @staticmethod
    def get_rgb_texture(uvp_rgb, stylized_images):
        # result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        result_views = stylized_images
        resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
        result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
        textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False)
        result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
        return result_tex_rgb, result_tex_rgb_output
    
    @torch.no_grad()
    def __call__(
        self,
        stylized_images
    ):
        self.uvp.to(self._execution_device)

        self.uvp_rgb.to(self._execution_device)
        result_tex_rgb, result_tex_rgb_output = self.get_rgb_texture(self.uvp_rgb, stylized_images.to(dtype=torch.float32))
        self.uvp.save_mesh(f"{self.result_dir}/textured.obj", result_tex_rgb.permute(1,2,0))


        self.uvp_rgb.set_texture_map(result_tex_rgb)
        textured_views = self.uvp_rgb.render_textured_views()
        textured_views_rgb = torch.cat(textured_views, axis=-1)[:-1,...]
        textured_views_rgb = textured_views_rgb.permute(1,2,0).cpu().numpy()[None,...]
        v = numpy_to_pil(textured_views_rgb)[0]
        v.save(f"{self.result_dir}/textured_views_rgb.jpg")
        # display(v)
        
        os.system(f'obj2gltf -i {self.result_dir}/textured.obj -o {self.result_dir}/textured.glb')

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        self.uvp.to("cpu")
        self.uvp_rgb.to("cpu")

        return textured_views_rgb[0]