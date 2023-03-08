config = {
    "version": "1.1.0",
    "batch_size": 512,
    "epochs": 100,
    "lr_gen": 2e-4,
    "lr_disc": 3e-4,
    "betas_disc": (0.9, 0.3),
    "betas_gen": (0.5, 0.35),
    "checkpoint": 5,
    "weight_checkpoint": 15,
    "weight_dir": ("trg1_disc.pt", "trg1_gen.pt"),
    "project_name": "TreeGAN_1.1.0"
}