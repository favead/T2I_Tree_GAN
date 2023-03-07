config = {
    "version": "1.0.0",
    "batch_size": 128,
    "epochs": 100,
    "gamma_gen": 0.9,
    "gamma_disc": 0.9,
    "time_step": 40,
    "lr_gen": 2e-4,
    "lr_disc": 2e-4,
    "betas": (0.9, 0.5),
    "checkpoint": 5,
    "weight_checkpoint": 15,
    "weight_dir": ("trg1_disc.pt", "trg1_gen.pt"),
    "project_name": "TreeGAN_1.0.0"
}