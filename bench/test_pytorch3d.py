from bench_suite import BenchSuite
from config import BenchConfig
import humanize as H
import numpy as np
import imageio.v2 as iio2
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesVertex,
)
from tensor2frame_bench.timer import Timer

RENDER_BATCH_SIZE = 32

config = BenchConfig()
suite = BenchSuite(
    config=config,
)
suite.prepare_geometry()

verts = suite.verts  # (frames, N, 3)
faces = suite.faces  # (M, 3)

# pytorch3d codes
mesh = Meshes(
    verts=torch.tensor(verts, dtype=torch.float32, device="cuda"),
    faces=torch.tensor(faces, dtype=torch.int64, device="cuda").expand(verts.shape[0], -1, -1),
)
mesh.textures = TexturesVertex(
    verts_features=torch.full(
        verts.shape, # type: ignore # (frames, N, 3)
        0.7,
        device="cuda",
    )
)

cam_R, cam_T = look_at_view_transform(
    dist=3,
    device="cuda",
)
cameras = FoVPerspectiveCameras(device="cuda", R=cam_R, T=cam_T)

raster_settings = RasterizationSettings(
    image_size=(config.height, config.width),
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    ),
    shader=HardPhongShader(
        device="cuda",
        cameras=cameras,
        lights=DirectionalLights(device="cuda"),
    )
)

def closure():
    writer = iio2.get_writer(
        'out.mp4',
        format='FFMPEG', # type: ignore
        mode='I', # multiple frames
        fps=config.fps,
    )
    with torch.inference_mode():
        for st in range(0, verts.shape[0], RENDER_BATCH_SIZE):
            ed = min(st + RENDER_BATCH_SIZE, verts.shape[0])
            
            frames = renderer(mesh[st:ed])
            for frame in range(frames.shape[0]):
                img = frames[frame, ..., :3].cpu().numpy()
                img = (img * 255).astype(np.uint8)
                writer.append_data(img)
    writer.close()

timer = Timer(func=closure, repeat=4)
timer.run()

# print results
print("FPS: ", H.metric(verts.shape[0] / timer.avg))
print("Average time per run: ", f"{timer.avg:.6f}s")
print("Records: ")
for i, r in enumerate(timer.records):
    print(f"  Run {i+1}: {r:.6f}s")