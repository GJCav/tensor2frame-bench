from alpenstock.settings import Settings
from pydantic import Field

class BenchConfig(Settings):
    width: int = Field(default=480, description="Width of the output frames.")
    height: int = Field(default=480, description="Height of the output frames.")
    
    mandelbulb_grid_size: int = Field(default=100, description="Grid size for the Mandelbulb fractal.")
    duration: float = Field(default=3.0, description="Duration of the generated video in seconds.")
    fps: int = Field(default=60, description="Frames per second of the generated video.")
    rotation_speed: float = Field(default=3.1415926, description="Rotation speed of the fractal animation.")
