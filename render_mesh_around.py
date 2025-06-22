"""
Sample code to render a cow with 360-degree views.

Usage:
 python render_mesh_around.py --image_size 512 --output_path ../outputs/abc --cow_path /home/khoahd/vdt_phase1/outputs/dreamgaussian/3e6572a1dcee4ab3b252a67a60d5d6f7_rgba.obj"""
import argparse
import imageio
import matplotlib.pyplot as plt
import pytorch3d
import torch

from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    TexturesUV
)
from pytorch3d.io import load_obj
from render_mesh import get_device, get_mesh_renderer, load_cow_mesh

def render_cow(
    cow_path="data/cow.obj", 
    image_size=256, 
    device=None,
    num_views=6,
):
    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)

    # Load mesh with texture
    verts, faces, aux = load_obj(
        cow_path, 
        load_textures=True,
    )
    faces_idx = faces.verts_idx.unsqueeze(0)
    verts = verts.unsqueeze(0)
    # Load texture map
    if aux.texture_images and len(aux.texture_images) > 0:
        tex_map = list(aux.texture_images.values())[0]
        tex_map = tex_map.unsqueeze(0)
        textures = TexturesUV(
            maps=tex_map,
            faces_uvs=faces.textures_idx.unsqueeze(0),
            verts_uvs=aux.verts_uvs.unsqueeze(0),
        )
    else:
        # fallback: white
        textures = None

    mesh = pytorch3d.structures.Meshes(
        verts=verts,
        faces=faces_idx,
        textures=textures,
    ).to(device)

    images = []
    view_params = [
        {"dist": 2, "elev": 0, "azim": 0},
        {"dist": 2, "elev": 0, "azim": 90},
        {"dist": 2, "elev": 0, "azim": 180},
        {"dist": 2, "elev": 0, "azim": 270},
        {"dist": 2, "elev": 90, "azim": 0},
        {"dist": 2, "elev": -90, "azim": 0},
    ]

    for params in view_params:
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=params["dist"],
            elev=params["elev"],
            azim=params["azim"]
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        # lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        lights = pytorch3d.renderer.AmbientLights(device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend.cpu().numpy()[0]
        images.append((image * 255).astype('uint8'))

    return images

# def save_gif(images, output_path, fps=15):
#     imageio.mimsave(output_path, images, fps=fps)
#     print("save successfully gif")

# e19c9fd36c40455b9d8d3ff6e44bd945_rgba.png
# 5de20229abd744a4920901126713a0f5_rgba.png
# 18abc857c9bf474090ddb1ec772a05db_rgba.png
# 44de128ebb974e95be8ad40bad273b57_rgba.png

file_names = [
    "e19c9fd36c40455b9d8d3ff6e44bd945_rgba.obj",
    "5de20229abd744a4920901126713a0f5_rgba.obj",
    "18abc857c9bf474090ddb1ec772a05db_rgba.obj",
    "44de128ebb974e95be8ad40bad273b57_rgba.obj"
    # "model.obj"
]

if __name__ == "__main__":

    # Input a folder arguments, generate 6 views for each file in the folder
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path_folder", type=str, default="data")
    parser.add_argument("--output_path", type=str, default="output/cow_rotation")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_views", type=int, default=200)
    # parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    for file_name in file_names:
        cow_path = f"{args.cow_path_folder}/{file_name}"
        print(f"Rendering {cow_path}...")
        images = render_cow(
            cow_path=cow_path, 
            image_size=args.image_size,
            num_views=args.num_views
        )

        name = file_name.split(".")[0]
        output_path = f"{args.output_path}/{name}"
        # Create output directory if it doesn't exist
        import os
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Save the images in the folder output path
        for i, img in enumerate(images):
            plt.imsave(f"{output_path}_view_{i}.png", img)
