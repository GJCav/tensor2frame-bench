from bench_suite import BenchSuite
from config import BenchConfig
import humanize as H
import pygfx as gfx
from rendercanvas.auto import RenderCanvas
from rendercanvas.offscreen import OffscreenRenderCanvas
import pylinalg as la
import numpy as np
import imageio.v3 as iio
import imageio.v2 as iio2

from tensor2frame_bench.timer import Timer


config = BenchConfig()
suite = BenchSuite(
    config=config,
)
suite.prepare_geometry()

verts = suite.verts
faces = suite.faces
vert_normals = suite.vert_normals

# pygfx codes

canvas = OffscreenRenderCanvas(
    size=(config.width, config.height),
    max_fps=0, # no limit
    vsync=False,
)
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()
scene.add(gfx.Background.from_color("white"))

scene.add(gfx.AmbientLight(intensity=0.8))

dir_light = gfx.DirectionalLight()
dir_light.local.position = (1, 0, 0) # type: ignore
dir_light.target.position = (0, 0, 0) # type: ignore
scene.add(dir_light)

fractal = gfx.Mesh(
    geometry=gfx.Geometry(
        positions=verts[0],
        indices=faces,
        normals=vert_normals[0],
    ),
    material=gfx.MeshPhongMaterial(color="gray")
)
scene.add(fractal)

camera = gfx.PerspectiveCamera(45, 1)
camera.show_object(fractal, view_dir=(-1, -1, -1))

# display = gfx.Display(
#     camera=camera,
# )
# display.show(scene)

def closure():
    writer = iio2.get_writer(
        'out.mp4', 
        format='FFMPEG', # type: ignore
        mode='I', # multiple frames
        fps=config.fps,
    )
    for frame in range(verts.shape[0]):
        fractal.geometry.positions.data[:] = verts[frame] # type: ignore
        fractal.geometry.positions.update_full() # type: ignore
        fractal.geometry.normals.data[:] = vert_normals[frame] # type: ignore
        fractal.geometry.normals.update_full() # type: ignore

        renderer.render(scene, camera)
        img = np.asarray(canvas.draw())
        writer.append_data(img)
    writer.close()

timer = Timer(func=closure)
timer.run()

# print results
print("----- pygfx + iio -----")
print("FPS: ", H.metric(verts.shape[0] / timer.avg))
print("Average time per run: ", f"{timer.avg:.6f}s")
print("Records: ")
for i, r in enumerate(timer.records):
    print(f"  Run {i+1}: {r:.6f}s")


# iio only for reference
images = []
for frame in range(verts.shape[0]):
    fractal.geometry.positions.data[:] = verts[frame] # type: ignore
    fractal.geometry.positions.update_full() # type: ignore
    fractal.geometry.normals.data[:] = vert_normals[frame] # type: ignore
    fractal.geometry.normals.update_full() # type: ignore

    renderer.render(scene, camera)
    img = np.asarray(canvas.draw())
    images.append(img)

def iio_closure():
    writer = iio2.get_writer(
        'out.mp4', 
        format='FFMPEG', # type: ignore
        mode='I', # multiple frames
        fps=config.fps,
    )
    for frame_idx in range(int(config.duration * config.fps)):
        img = images[frame_idx]
        writer.append_data(img)
    writer.close()
    
timer = Timer(func=iio_closure)
timer.run()

print("----- iio only -----")
print("FPS: ", H.metric(verts.shape[0] / timer.avg))
print("Average time per run: ", f"{timer.avg:.6f}s")
print("Records: ")
for i, r in enumerate(timer.records):
    print(f"  Run {i+1}: {r:.6f}s")
