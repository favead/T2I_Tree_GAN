from typing import Any, List, Tuple, Callable, Dict
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configure when developing new model
    """
    version: str
    batch_size: int
    epochs: int
    gamma: float
    lr: float
    betas: Tuple[float]
    train_file: str
    val_file: str
    weight_dir: Tuple[str, str]
    test_data_paths: Tuple[str, str]
    train_data_path: Tuple[str, str]