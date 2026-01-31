from bench_suite import BenchSuite
from config import BenchConfig
import humanize as H

from tensor2frame_bench.timer import Timer
from tensor2frame_bench.pyvista_func import PyVistaSuite

config = BenchConfig()
suite = BenchSuite(
    config=config,
)
suite.prepare_geometry()

verts = suite.verts
faces = suite.faces

s = PyVistaSuite(
    verts=verts,
    faces=faces,
    width=config.width,
    height=config.height,
    offscreen=True,
    fps=config.fps,
)
s.initialize()
s.show()

timer = Timer(
    func=lambda: s.render_to_video("./out.mp4"),
)
timer.run()

s.close()


# print results
print("FPS: ", H.metric(verts.shape[0] / timer.avg))
print("Average time per run: ", f"{timer.avg:.6f}s")
print("Records: ")
for i, r in enumerate(timer.records):
    print(f"  Run {i+1}: {r:.6f}s")