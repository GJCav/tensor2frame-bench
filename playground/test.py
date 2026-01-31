import pyvista as pv
import wgpu
from pprint import pprint

print("PyVista version:", pv.__version__)
print(pv.Report())

print("\n--------------- WGPU INFO ---------------")
pprint(wgpu.gpu.request_adapter_sync().info)
print("----------------------------------------\n")