config = {
    "version": "2.0.1",
    "batch_size": 256,
    "epochs": 60,
    "lr_gen": 3e-5,
    "lr_disc": 3e-5,
    "betas_disc": (0.5, 0.999),
    "betas_gen": (0.5, 0.999),
    "checkpoint": 1,
    "weight_checkpoint": 15,
    "weight_dir": ("trg2_0_0_disc.pt", "trg2_0_0_gen.pt"),
    "project_name": "TreeGAN_2.0.1"
}