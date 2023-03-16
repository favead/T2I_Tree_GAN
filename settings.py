config = {
    "version": "1.0.0",
    "batch_size": 256,
    "epochs": 40,
    "lr_gen": 3e-5,
    "lr_disc": 3e-5,
    "betas_disc": (0.5, 0.999),
    "betas_gen": (0.5, 0.999),
    "checkpoint": 1,
    "weight_checkpoint": 20,
    "weight_dir": ("t2itree_disc.pt", "t2itree_gen.pt"),
    "project_name": "T2ITreeGAN_1.0.0"
}