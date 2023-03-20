from typing import Tuple, Dict, Callable, Union
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


def iter_train(x: Tensor, sent_emb: Tensor, wrong_emb: Tensor, y:Tensor,
               model: Dict[str, nn.Module], optim: Dict[str, Optimizer],
               device: torch.device, loss_func: Dict[str, Callable],
               wandb: object, gen_iter: int, disc_iter: int) -> None:
    gen_emb = torch.hstack((x, sent_emb)).to(device)
    disc_emb = sent_embed.view(sent_embed.size()[0], 48, 4, 4)
    for _ in range(disc_iter):
        with torch.set_grad_enabled(False):
            gen = model["generator"](gen_emb)
        sr = model["discriminator"](y, disc_emb)
        sw = model["discriminator"](y, wrong_emb)
        sf = model["discriminator"](gen, disc_emb)
        loss_d = loss_func["discriminator"](sr, sw, sf, device)

        optim["discriminator"].zero_grad()
        loss_d.backward()
        optim["discriminator"].step()

    
    for _ in range(gen_iter):
        gen = model["generator"](gen_emb)
        sf = model["discriminator"](gen, disc_emb)
        loss_g = loss_func["generator"](sf, device)

        optim["generator"].zero_grad()
        loss_g.backward()
        optim["generator"].step()

    wandb.log({"generator_loss": loss_g.item(), "discriminator_loss": loss_d.item()})
    return None


def train(model: Dict[str, nn.Module], dataset: Dataset, optim: Dict[str, Optimizer],
          scheduler: Dict[str, StepLR], loss_func: Dict[str, Callable],
          wght_dir: Tuple[str, str], batch_size: int, epochs: int, wandb: object,
          device: torch.device, tqdm, config: Dict[str, Union[str, int, float]],
          tensor2image: Callable, rgb2srgb: Callable, fake_const: Tensor,
          get_embedding: Callable, get_wrong_embedding: Callable, gc=None) -> None:
    dloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=2)
    for epoch in tqdm(range(1, epochs + 1)):
        wandb.init(
            project=f"{config['project_name']}_train",
            name=f"epoch_{epoch}",
            config=config)
        for i, y in tqdm(enumerate(dloader)):
            x = torch.randn(batch_size, 100, 1, 1)
            sent_emb = get_embedding(i)
            wrong_emb = get_wrong_embedding(i)
            x, sent_emb, wrong_emb, y = x.to(device), sent_emb.to(device), wrong_emb.to(device), y.to(device)
            iter_train(x, sent_emb, wrong_emb, y, model, optim, device, loss_func, wandb, config['gen_iter'],
            config['disc_iter'])
        if gc:
            gc.collect()
        if scheduler:
            scheduler["discriminator"].step()
            scheduler["generator"].step()
        if (epoch % config["checkpoint"]) == 0:
            gen = (predict_one_sample(model["generator"], fake_const, device, tensor2image))
            wandb.log({"image": wandb.Image(gen)})
        if (epoch % config["weight_checkpoint"]) == 0:
            torch.save(copy.deepcopy(model["discriminator"].state_dict()), f"{epoch}_ep_{wght_dir[0]}")
            torch.save(copy.deepcopy(model["generator"].state_dict()), f"{epoch}_ep_{wght_dir[1]}")
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
