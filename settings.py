config = {
    "version": "1.0.2",
    "batch_size": 256,
    "epochs": 100,
    "gamma_gen": 0.9,
    "gamma_disc": 0.9,
    "time_step": 40,
    "lr_gen": 1e-5,
    "lr_disc": 2e-6,
    "betas": (0.9, 0.5),
    "checkpoint": 5,
    "weight_checkpoint": 15,
    "weight_dir": ("trg1_disc.pt", "trg1_gen.pt"),
    "project_name": "TreeGAN_1.0.2"
}