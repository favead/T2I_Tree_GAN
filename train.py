from typing import List, Tuple, Dict, Callable, Union
import copy
import torch
import cv2
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
from skimage.metrics import structural_similarity
import numpy as np

def predict_one_sample(model: nn.Module, lr: Tensor, device: torch.device,
                       tensor2image: Callable) -> np.ndarray:
    model.eval()
    with torch.set_grad_enabled(False):
        x = lr.to(device)
        output = model(x.unsqueeze(0))
        output = output.cpu()
    out = tensor2image(output, normalize=True, only_y=False)
    return out


def train(model: Dict[str, nn.Module], dataset: Dataset, optim: Dict[str, Optimizer],
          scheduler: Dict[str, StepLR], loss_func: Dict[str, Callable], metric: Callable,
          wght_dir: Tuple[str, str], batch_size: int, epochs: int, wandb: object,
          device: torch.device, tqdm, vgg_modules: List[nn.Module], config: Dict[str, Union[str, int, float]],
          AverageMeter: object, val_dataset: Dataset, tensor2image: Callable,
          gc=None, save_weights: bool = True, is_init: bool = False) -> None:
    dloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    best_weights = {"generator": copy.deepcopy(model["generator"].state_dict()),
                    "discriminator": copy.deepcopy(model["discriminator"].state_dict())}
    best_disc_loss = np.inf
    best_metric = 0.0
    for epoch in tqdm(range(1, epochs + 1)):
        if not is_init:
            wandb.init(
                project=f"{config['version']}_train",
                name=f"epoch_{epoch}",
                config=config)
        disc_meter = AverageMeter()
        gen_psnr_meter = AverageMeter()
        for x, y in dloader:
            lr, hr = x.to(device), y.to(device)
            
            ##########_Discriminator iteration_########
            
            with torch.set_grad_enabled(False):
                sr = model["generator"](lr)
            sr_labels, hr_labels = torch.zeros(len(sr), 1).to(device), torch.ones(len(hr), 1).to(device)
            hr_preds = model["discriminator"](hr)
            sr_preds = model["discriminator"](sr)
            hr_loss = loss_func["discriminator"](hr_preds, hr_labels)
            sr_loss = loss_func["discriminator"](sr_preds, sr_labels)
            disc_loss = (hr_loss + sr_loss) / 2.
            
            optim["discriminator"].zero_grad()
            disc_loss.backward()
            optim["discriminator"].step()
            
            ########_Generator iteration_###############
            
            sr = model["generator"](lr)
            sr_labels = torch.ones(len(hr), 1).to(device)
            sr_preds = model["discriminator"](sr)
            vgg_loss, pixel_loss, adv_loss = loss_func["generator"](vgg_modules, sr, hr, sr_preds, sr_labels,
                                                                    device, config)
            loss = 0.006 * vgg_loss + 1e-2 * adv_loss + pixel_loss
            optim["generator"].zero_grad()
            loss.backward()
            optim["generator"].step()
            
            ########_Update losses and metrics_#############
            gen_psnr = metric(sr.cpu(), hr.cpu()).item()
            disc_meter.update(disc_loss.item(), len(lr))
            gen_psnr_meter.update(gen_psnr, len(sr))
            
            wandb.log({"generator_vgg_loss": vgg_loss, "generator_pixel_loss": pixel_loss,
                       "generator_adv": adv_loss, "discriminator_loss": disc_loss,
                       "generator_loss": loss, "generator_psnr": gen_psnr})
        if gc:
            gc.collect()
        gen_psnr = gen_psnr_meter.avg
        scheduler["discriminator"].step()
        scheduler["generator"].step()
        if  disc_meter.avg > best_disc_loss:
            best_disc_loss = disc_meter.avg
            best_weights["discriminator"] = copy.deepcopy(model["discriminator"].state_dict())
        if  gen_psnr_meter.avg > best_metric:
            best_metric = gen_psnr_meter.avg
            best_weights["generator"] = copy.deepcopy(model["generator"].state_dict())
        if val_dataset:
            table = wandb.Table(columns=["NN", "GT", "PSNR", "SSIM"])
            lr, hr = val_dataset[0]
            nn_img = predict_one_sample(model["generator"], lr, device, tensor2image)
            psnr = cv2.PSNR(nn_img, hr)
            ssim = structural_similarity(nn_img, hr, channel_axis=2,
                                                multichannel=True)
            table.add_data(wandb.Image(nn_img), wandb.Image(hr), psnr, ssim)
            wandb.log({"eval_epoch":table}, commit=False)
        if not is_init:
            wandb.finish()
    if save_weights:
        torch.save(best_weights["generator"], wght_dir[1])
        torch.save(best_weights["discriminator"], wght_dir[0])
        wandb.init(project=f"{config['project_name']}_weights")
        wandb.save(wght_dir[1])
        wandb.save(wght_dir[0])
        wandb.finish()
    return None


def predict(model: nn.Module, test_dataset: Dataset, device: torch.device,
            tensor2image: Callable) -> np.ndarray:
    model.eval()
    images = []
    out_images = []
    for i in range(len(test_dataset)):
        lr, hr_img = test_dataset[i]
        out = predict_one_sample(model, lr, device, tensor2image)
        out_images.append(out)
        images.append(hr_img)
    return out_images, images
