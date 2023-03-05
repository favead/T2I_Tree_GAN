import os
from typing import List, Module
import torch
from torch import Tensor
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from settings import Config
from imageproc import read_image, resize_image, rgb2srgb



class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_psnr(img1: Tensor, img2: Tensor) -> float:
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def log_image_table(out_images: List[np.ndarray], gt: List[np.ndarray],
                    filenames: List[str], name: str, wandb: Module, 
                    scale: int = Config.scale,) -> None:
    wandb.init(
        project=f"{Config.project_name}_test",
        name=name,
        config={
            "epochs": Config.epochs,
            "batch_size": Config.batch_size,
            "lr": Config.lr_gen
            })
    table = wandb.Table(columns=["linear", "cubic", "NN", "gt",
                                 "Filename", "PSNR", "SSIM", "NN/LIN PSNR",
                                 "NN/LIN SSIM", "NN/CUBIC PSNR", "NN/CUBIC SSIM"])
    for i in range(len(out_images)):
        lr_p, hr_p = filenames[i]
        lr_img = read_image(lr_p)
        lin_img = resize_image(lr_img, scale, is_up=True, typ=cv2.INTER_LINEAR)
        cub_img = resize_image(lr_img, scale, is_up=True, typ=cv2.INTER_CUBIC)
        nn_psnr = cv2.PSNR(out_images[i], gt[i])
        lin_psnr = cv2.PSNR(lin_img, gt[i])
        cub_psnr = cv2.PSNR(cub_img, gt[i])
        nn_ssim = structural_similarity(out_images[i], gt[i], channel_axis=2,
                                             multichannel=True)
        lin_ssim = structural_similarity(lin_img, gt[i], channel_axis=2,
                                             multichannel=True)
        cub_ssim = structural_similarity(cub_img, gt[i], channel_axis=2,
                                     multichannel=True)
        table.add_data(wandb.Image(rgb2srgb(lin_img)),
                       wandb.Image(rgb2srgb(cub_img)),
                       wandb.Image(rgb2srgb(out_images[i])),
                       wandb.Image(rgb2srgb(gt[i])),
                       str(os.path.basename(lr_p)),
                       nn_psnr,
                       nn_ssim,
                       nn_psnr / lin_psnr,
                       nn_ssim / lin_ssim,
                       nn_psnr / cub_psnr,
                       nn_ssim / cub_ssim)
    wandb.log({"benchmark_set14":table}, commit=False)
    wandb.finish()
    return None