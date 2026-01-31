import pyvista as pv
import numpy as np
from attrs import define, field


@define(kw_only=True)
class PyVistaSuite:
    verts: np.ndarray = field()  # shape (frames, N, 3)
    faces: np.ndarray = field()  # shape (M, 3)
    
    width: int = field(default=800)
    height: int = field(default=800)
    fps: int = field()
    offscreen: bool = field(default=False)
    
    pl: pv.Plotter | None = field(init=False, default=None)
    mesh: pv.PolyData | None = field(init=False, default=None)
    
    def initialize(self):
        self.pl = pv.Plotter(window_size=[self.width, self.height], off_screen=self.offscreen)
        self.mesh = pv.PolyData(self.verts[0], np.hstack((np.full((self.faces.shape[0], 1), 3), self.faces)))
        self.pl.add_mesh(self.mesh, color='gray')
    
    def show(self, cpos=None):
        if self.pl is None:
            raise RuntimeError("Call initialize() before showing.")
        self.pl.show(
            interactive=False, # Disable user interaction
            interactive_update=True, # Show in non-blocking mode
            cpos=cpos
        )
        
    def close(self):
        if self.pl is None:
            raise RuntimeError("Call initialize() before closing.")
        self.pl.close()

    def render_to_window(self):
        if self.pl is None:
            raise RuntimeError("Call initialize() and show() before rendering.")
        
        self.pl.render()  # In offscreen mode, this creates the rendering context
        
        for frame in range(self.verts.shape[0]):
            self.mesh.points = self.verts[frame] # type: ignore
            self.pl.update()
            
    def render_loop_only(self):
        """
        Loop through frames without rendering, for benchmarking.
        """
        if self.pl is None:
            raise RuntimeError("Call initialize() before looping.")
        for frame in range(self.verts.shape[0]):
            self.mesh.points = self.verts[frame] # type: ignore
    
    def render_to_video(self, filename: str):
        """
        Set `filename` to a ramdisk path to eliminate disk I/O impact.
        """
        pl = self.pl
        if pl is None:
            raise RuntimeError("Call initialize() before rendering to video.")
        
        pl.open_movie(filename, framerate=self.fps)
        pl.write_frame()  # Write initial frame
        for frame in range(self.verts.shape[0]):
            self.mesh.points = self.verts[frame]  # type: ignore
            pl.write_frame()  # write_frame implicitly calls update()
        