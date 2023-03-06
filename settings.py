config = {
    "version": "1.1.3",
    "batch_size": 64,
    "crop_size": 9216,
    "epochs": 10,
    "scale": 2,
    "gamma_gen": 0.9,
    "crop_times": 16,
    "gamma_disc": 0.9,
    "time_step": 4,
    "lr_gen": 1e-4,
    "lr_disc": 1e-4,
    "betas": (0.9, 0.75),
    "weight_dir": ("ch9_disc.pt", "ch9_gen.pt"),
    "crop_times": 16,
    "crop_area": 9216,
    "vgg_layers": (2, 5, 9, 12, 16, 19, 22),
    "project_name": "SRGAN_1.1.3"
}