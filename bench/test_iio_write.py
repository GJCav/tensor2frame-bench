from config import BenchConfig
import numpy as np
import imageio.v2 as iio2
import humanize as H
from tensor2frame_bench.timer import Timer

config = BenchConfig()

def random_image():
    return (np.random.rand(config.height, config.width, 3) * 255).astype(np.uint8)

images = []
for i in range(int(config.duration * config.fps)):
    img = random_image()
    images.append(img)

def closure():
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

timer = Timer(func=closure)
timer.run()

# print results
print("FPS: ", H.metric(config.duration * config.fps / timer.avg))
print("Average time per run: ", f"{timer.avg:.6f}s")
print("Records: ")
for i, r in enumerate(timer.records):
    print(f"  Run {i+1}: {r:.6f}s")