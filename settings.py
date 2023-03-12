config = {
    "version": "2.5.0",
    "batch_size": 256,
    "epochs": 81,
    "lr_gen": 2e-4,
    "lr_disc": 2e-4,
    "betas_disc": (0.5, 0.999),
    "betas_gen": (0.5, 0.999),
    "checkpoint": 1,
    "weight_checkpoint": 20,
    "weight_dir": ("trg2_5_0_disc.pt", "trg2_5_0_gen.pt"),
    "project_name": "TreeGAN_2.5.0"
}