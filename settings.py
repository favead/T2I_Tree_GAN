config = {
    "version": "1.2.0",
    "batch_size": 256,
    "epochs": 100,
    "lr_gen": 2e-4,
    "lr_disc": 1e-5,
    "betas_disc": (0.85, 0.4),
    "betas_gen": (0.4, 0.35),
    "checkpoint": 5,
    "weight_checkpoint": 15,
    "weight_dir": ("trg1_2_disc.pt", "trg1_2_gen.pt"),
    "project_name": "TreeGAN_1.2.0"
}