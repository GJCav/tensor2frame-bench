# tensor2frame-bench

**tensor2frame-bench** evaluates the rasterization performance and developer experience of common Python 3D rendering libraries for AI research workflows.

## Overview

In deep learning research, visualizing 3D model outputs (such as heavy meshes or point clouds) and saving them to video is a common yet secondary task. Researchers often need a solution that balances high performance with minimal setup time.

This benchmark compares libraries like **PyVista**, **PyGfx**, and **PyTorch3D** in a specific "tensor-to-frame" scenario:
1.  **Stress Test**: Rendering a complex, procedurally generated Mandelbulb fractal to stress the rasterizer.
2.  **Primary Metric**: Frames Per Second (FPS) when rendering to a video file.
3.  **Secondary Metric**: Developer friendliness and ease of use.

The goal is to identify which tools allow authors to generate high-quality visualizations efficiently without distracting from their core research.


## FPS Results

All results were obtained on a machine with:

- System: Ubuntu 22.04 LTS Desktop
- CPU: AMD Ryzen 9 9950X (8 vCPUs, virtualized by KVM/QEMU/libvirt)
- RAM: 8 GB DDR5 (pinned)
- GPU: NVIDIA GTX 1080 Ti (passthrough by VFIO, driver version 580)
- CUDA: 11.8 (limited as newer CUDA no longer supports GTX 1080 Ti)

The generated Mandelbulb mesh has:

- Vertices: ~ 108 k
- Faces: ~ 213 k

This is a relatively low stress test for modern GPUs, but it reflects common research scenarios where models are not excessively complex.

| Configuration | FPS |
| --- | --- |
| PyVista (CPU) | 26.5 |
| PyVista (GPU) | 183 |
| PyGfx | 278 (best) |
| Pytorch3d | 8.58 (worst) |

*Notes*:

- `PyVista (CPU)` uses software rendering via *Mesa*, *llvmpipe* and `vtkXOpenGLRenderWindow` because the test was conducted in the VNC.
- `PyVista (GPU)` uses the GPU with `vtkEGLRenderWindow`, as the test was conducted from an SSH session.
- `PyGfx` uses the GPU, too.
- `Pytorch3d` features the differentiable rasterizer, which is not ideal for pure visualization tasks, so its performance is the lowest. But I find researchers often use it for that purpose nonetheless.
- Video writer performance: all tests use `imageio` with FFMPEG to write MP4 files. The video writing overhead is minimal compared to rendering time. As a reference, the pure video writing speeds is ~800 FPS in the `PyGfx` test.

## Developer Experience

**PyVista**: 
- Comment: Very easy to set up and use. 
- Doc: Though the doc is a bit scattered, the straightforward API makes it easy to figure out how to render and save videos.
- Pros: 
    - Yields decent performance with minimal code.
    - Uses VTK under the hood, which is mature and provides VTK-based export options (`vtp` files) for finer visuzalization and inspection with `ParaView`.
- Cons: 
    - It may silently fall back to CPU rendering without warning. 
    - The extendability is limited by the VTK backend.

**PyGfx**:
- Comment: Performant but young, currently in beta stage, not recommended.
- Doc: Unclear, sparse, and less organized. Requires digging into source code, trials and errors, and digging to related projects (e.g., `rendercanvas`, `wgpu-py`, etc.) to figure out how to use it effectively.
- Pros:
    - Best performance in the benchmark.
    - Modern API design (WebGPU-based).
- Cons:
    - Immature library, possibly breaking changes in the future.
    - Almost no typing hints. Every property/attribute is implemented by `get` and `set`, making it hard to figure out what attributes are available.
    - API design has ambiguities and docs do not yet clarify them.

**Pytorch3d**:
- Comment: Designed for differentiable rendering, absolutely not for pure visualization tasks.
- Doc: Good documentation, with examples and tutorials.
- Pros:
    - Well-integrated with PyTorch ecosystem.
    - Good for differentiable rendering tasks.
- Cons:
    - Poor performance for non-differentiable rendering tasks.
    - High GPU memory consumption.

**ModernGL**
- Not tested!
- Pros:
    - Export native OpenGL commands, giving full control over rendering pipeline.
- Cons:
    - Requires writing shaders in GLSL manually, which is cumbersome for common tasks.

## Conclusion

- For visualization tasks, choose **PyVista** for ease of use and decent performance.
- Keep an eye on **PyGfx** as it matures, as it shows promise for high performance.
- Avoid using **Pytorch3d** for visualization, even previous pipelines may have used it for AI training. It is too slow and consumes huge GPU memory.