from typing import Any, List, Tuple, Callable, Dict

config = {
    "version": "1.1.1",
    "batch_size": 64,
    "crop_size": 9216,
    "epochs": 10,
    "scale": 2,
    "gamma_gen": 0.8,
    "gamma_disc": 0.8,
    "time_step": 4,
    "lr_gen": 2e-4,
    "lr_disc": 2e-4,
    "betas": (0.9, 0.5),
    "weight_dir": ("ch9_disc.pt, ch9_gen.pt"),
    "crop_times": 16,
    "crop_area": 9216,
    "vgg_layers": (2, 5, 9, 12, 16, 19, 22),
    "project_name": "SRGAN_1.1.1"
}