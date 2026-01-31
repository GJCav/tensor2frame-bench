from attrs import define, field
from config import BenchConfig
import numpy as np
import humanize as H
import torch
import roma

import tensor2frame_bench.fractals as fractals


@define(kw_only=True)
class BenchSuite:
    config: BenchConfig = field()
    
    verts: np.ndarray = field(init=False, default=None) # (frames, N, 3)
    faces: np.ndarray = field(init=False, default=None) # (M, 3)
    vert_normals: np.ndarray = field(init=False, default=None) # (frames, N, 3)
    
    def prepare_geometry(self):
        g = fractals.Mandelbulb(grid_size=self.config.mandelbulb_grid_size)
        
        print("Vert count: ", H.metric(g.vertices.shape[0]))
        print("Face count: ", H.metric(g.faces.shape[0]))


        # rotate around Y axis
        rot_angles = (
            torch.linspace(0, self.config.duration, int(self.config.fps * self.config.duration), device="cpu")
            * self.config.rotation_speed
        )
        axis = torch.tensor([0, 1, 0], dtype=torch.float32, device="cpu")
        rotations = axis.unsqueeze(0) * rot_angles.unsqueeze(1)
        rot_mats = roma.rotvec_to_rotmat(rotations)  # (frames, 3, 3)
        rot_mats = rot_mats.permute(0, 2, 1)

        # apply rotations to verts for each frame
        verts = torch.tensor(g.vertices, dtype=torch.float32, device="cpu")  # (N, 3)
        verts = verts @ rot_mats  # (frames, N, 3)
        
        # apply rotations to vertex normals for each frame
        vert_normals = torch.tensor(g.vert_normals, dtype=torch.float32, device="cpu")  # (N, 3)
        vert_normals = vert_normals @ rot_mats  # (frames, N, 3)

        print(
            "verts size for all frames: ",
            H.naturalsize(verts.element_size() * verts.nelement()),
        )

        # convert to numpy and render with pyvista
        verts = verts.cpu().numpy()  # (frames, N, 3)
        faces = g.faces  # (M, 3)
        vert_normals = vert_normals.cpu().numpy()  # (frames, N, 3)
        
        # store
        self.verts = verts
        self.faces = faces
        self.vert_normals = vert_normals
