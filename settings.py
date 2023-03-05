from typing import Any, List, Tuple, Callable, Dict
from dataclasses import dataclass


@dataclass
class Config:
    version: str = "1.1.0"
    batch_size: int = 64
    crop_size: int = 9216
    epochs: int = 10
    scale: int = 2
    gamma_gen: float = 0.8
    gamma_disc: float = 0.8
    time_step: int = 4
    lr_gen: float = 2e-4
    lr_disc: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.5)
    weight_dir: Tuple[str, str] = ("disc.pt, gen.pt")