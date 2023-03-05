from typing import List, Tuple, Dict, Callable
import copy
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
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
          device: torch.device, tqdm, vgg_modules: List[nn.Module], Config: dict,
          AverageMeter: object) -> None:
    dloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    losses = {"ep_generator_vgg_loss":[], "ep_generator_pixel_loss": [], "ep_generator_adv": [],
              "ep_discriminator_loss":[], "ep_generator_loss": [], "ep_generator_psnr": []}
    best_weights = {"generator": copy.deepcopy(model["generator"].state_dict()),
                    "discriminator": copy.deepcopy(model["discriminator"].state_dict())}
    best_disc_loss = np.inf
    best_metric = 0.0
    for epoch in tqdm(range(1, epochs + 1)):
        wandb.init(
            project=f"{Config.version}_train",
            name=f"epoch_{epoch}",
            config={
                "epochs": Config.epochs,
                "batch_size": Config.batch_size,
                "lr": Config.lr_gen
        })
        disc_meter = AverageMeter()
        gen_meter = AverageMeter()
        gen_vgg_meter = AverageMeter()
        gen_pixel_meter = AverageMeter()
        gen_adv_meter = AverageMeter()
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
            vgg_loss, pixel_loss, adv_loss = loss_func["generator"](vgg_modules, sr, hr, sr_preds, sr_labels)
            loss = vgg_loss + 1e-2 * adv_loss + pixel_loss
            optim["generator"].zero_grad()
            loss.backward()
            optim["generator"].step()
            
            ########_Update losses and metrics_#############
            gen_psnr = metric(sr.cpu(), hr.cpu()).item()
            disc_meter.update(disc_loss.item(), len(lr))
            gen_psnr_meter.update(gen_psnr, len(sr))
            gen_meter.update(loss.item(), len(sr))
            gen_vgg_meter.update(vgg_loss, len(sr))
            gen_pixel_meter.update(pixel_loss, len(sr))
            gen_adv_meter.update(adv_loss, len(sr))
            
            wandb.log({"generator_vgg_loss": vgg_loss, "generator_pixel_loss": pixel_loss,
                       "generator_adv": adv_loss, "discriminator_loss": disc_loss,
                       "generator_loss": loss, "generator_psnr": gen_psnr})
        wandb.finish()
        gen_psnr = gen_psnr_meter.avg
        scheduler["discriminator"].step()
        scheduler["generator"].step()
        losses["ep_discriminator_loss"].append(disc_meter.avg)
        losses["ep_generator_psnr"].append(gen_psnr_meter.avg)
        losses["ep_generator_vgg_loss"].append(gen_vgg_meter.avg)
        losses["ep_generator_pixel_loss"].append(gen_pixel_meter.avg)
        losses["ep_generator_adv"].append(gen_adv_meter.avg)
        losses["ep_generator_loss"].append(gen_meter.avg)
        print(losses)
        if losses["ep_discriminator_loss"][-1] > best_disc_loss:
            best_disc_loss = losses["ep_discriminator_loss"][-1]
            best_weights["discriminator"] = copy.deepcopy(model["discriminator"].state_dict())
        if losses["ep_generator_psnr"][-1] > best_metric:
            best_metric = losses["ep_generator_psnr"][-1]
            best_weights["generator"] = copy.deepcopy(model["generator"].state_dict())
    torch.save(best_weights["generator"], wght_dir[1])
    torch.save(best_weights["discriminator"], wght_dir[0])
    return None


def predict(model: nn.Module, test_dataset: Dataset) -> np.ndarray:
    model.eval()
    images = []
    out_images = []
    for i in range(len(test_dataset)):
        lr, hr_img = test_dataset[i]
        out = predict_one_sample(model, lr)
        out_images.append(out)
        images.append(hr_img)
    return out_images, images
