# Script to generate Sierpinski Tetrahedron mesh

import torch
import numpy as np
import skimage.measure
from attrs import define, field


@define(kw_only=True)
class Mandelbulb:
    """Generate a Mandelbulb fractal mesh for 3D rendering.

    Control `grid_size` to adjust the resolution of the generated mesh
    so that higher values yield finer details and thus larger meshes.
    """
    
    grid_size: int = field(default=100)  # 200 = standard, 400+ = high res
    power: float = field(default=8.0)  # fractal complexity
    max_iteration: int = field(default=8)  # number of iterations
    lower_bound: float = field(default=-1)
    upper_bound: float = field(default=1)
    device: str = field(default="cuda")

    _verts: np.ndarray = field(init=False, default=None)  # shape (N, 3)
    _faces: np.ndarray = field(init=False, default=None)  # shape (M, 3)
    _vert_normals: np.ndarray = field(init=False, default=None)  # shape (N, 3)
    _values: np.ndarray = field(init=False, default=None)  # shape (N,)
    
    @property
    def vertices(self) -> np.ndarray:
        if self._verts is None:
            self._generate()
        return self._verts
    
    @property
    def faces(self) -> np.ndarray:
        if self._faces is None:
            self._generate()
        return self._faces
    
    @property
    def vert_normals(self) -> np.ndarray:
        if self._vert_normals is None:
            self._generate()
        return self._vert_normals

    def _generate(self):
        range_t = torch.linspace(
            self.lower_bound,
            self.upper_bound,
            self.grid_size,
            device=self.device,
            dtype=torch.float32,
        )
        volume = np.zeros(
            (self.grid_size, self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        
        grid_y, grid_x = torch.meshgrid(range_t, range_t, indexing='ij')

        with torch.no_grad():
            for i, z_val in enumerate(range_t):
                z_slice = torch.full_like(grid_x, z_val) # type: ignore
                sdf_slice = self._mandelbulb_iterate(
                    grid_x, grid_y, z_slice
                )
                volume[i, :, :] = sdf_slice.cpu().numpy()
        
        
        verts, faces, vert_normals, values = skimage.measure.marching_cubes(volume, level=2.4)
        
        # fix normal direction
        vert_normals = -vert_normals

        scale = (self.upper_bound - self.lower_bound) / self.grid_size
        verts = verts * scale + self.lower_bound
        
        self._verts = np.ascontiguousarray(verts)
        self._faces = np.ascontiguousarray(faces)
        self._vert_normals = np.ascontiguousarray(vert_normals)
        self._values = np.ascontiguousarray(values)
        
        
    def _mandelbulb_iterate(self, x, y, z):
        zx, zy, zz = x.clone(), y.clone(), z.clone()
        cx, cy, cz = x.clone(), y.clone(), z.clone()
        r = torch.zeros_like(x)
        
        for i in range(self.max_iteration):
            r = torch.sqrt(zx**2 + zy**2 + zz**2)
            mask = r < 2.0
            if not mask.any(): 
                break
            
            theta = torch.atan2(torch.sqrt(zx**2 + zy**2), zz)
            phi = torch.atan2(zy, zx)
            
            zr = torch.pow(r, self.power)
            theta = theta * self.power
            phi = phi * self.power
            
            zx = zr * torch.sin(theta) * torch.cos(phi) + cx
            zy = zr * torch.sin(theta) * torch.sin(phi) + cy
            zz = zr * torch.cos(theta) + cz
            
            # Avoid NaNs for diverged points
            zx[~mask] = 2.5
            zy[~mask] = 2.5
            zz[~mask] = 2.5

        return torch.sqrt(zx**2 + zy**2 + zz**2)
