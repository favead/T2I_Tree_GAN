config = {
    "version": "1.0.6",
    "batch_size": 256,
    "epochs": 30,
    "lr_gen": 1e-5,
    "lr_disc": 1e-5,
    "betas_disc": (0.5, 0.999),
    "betas_gen": (0.5, 0.999),
    "checkpoint": 1,
    "weight_checkpoint": 20,
    'gen_iter': 2,
    'disc_iter': 1,
    "weight_dir": ("t2itree_disc.pt", "t2itree_gen.pt"),
    "project_name": "T2ITreeGAN_1.0.6"
}
