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


def iter_disc(x: Tensor, y:Tensor, model: Dict[str, nn.Module], optim: Optimizer, device: torch.device,
              loss_func: Callable) -> float:
    with torch.set_grad_enabled(False):
        gen = model["generator"](x)
    gen_labels, y_labels = torch.zeros(len(gen), 1).to(device), torch.ones(len(y), 1).to(device)
    gen_preds, y_preds = model["discriminator"](gen), model["discriminator"](y)
    gen_loss, y_loss = loss_func(gen_preds, gen_labels), loss_func(y_preds, y_labels)
    disc_loss = (gen_loss + y_loss) / 2.
    optim.zero_grad()
    disc_loss.backward()
    optim.step()
    return disc_loss.item()


def iter_gen(x: Tensor, y:Tensor, model: Dict[str, nn.Module], optim: Optimizer, device: torch.device,
              loss_func: Callable) -> float:
    gen = model["generator"](x)
    gen_labels = torch.ones(len(y), 1).to(device)
    gen_preds = model["discriminator"](gen)
    gen_loss = loss_func(gen_preds, gen_labels)
    optim.zero_grad()
    gen_loss.backward()
    optim.step()
    return gen_loss.item()


def iter_train(x: Tensor, y:Tensor, model: Dict[str, nn.Module], optim: Dict[str, Optimizer], 
               device: torch.device, loss_func: Dict[str, Callable], wandb: object) -> None:
    disc_loss = iter_disc(x, y, model, optim["discriminator"], device, loss_func["discriminator"])
    gen_loss = iter_gen(x, y, model, optim["generator"], device, loss_func["generator"])
    wandb.log({"generator_loss": gen_loss, "discriminator_loss": disc_loss})
    return None


def train(model: Dict[str, nn.Module], dataset: Dataset, optim: Dict[str, Optimizer],
          scheduler: Dict[str, StepLR], loss_func: Dict[str, Callable],
          wght_dir: Tuple[str, str], batch_size: int, epochs: int, wandb: object,
          device: torch.device, tqdm, config: Dict[str, Union[str, int, float]],
          tensor2image: Callable, rgb2srgb: Callable, fake_const: Tensor, gc=None) -> None:
    dloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    for epoch in tqdm(range(1, epochs + 1)):
        wandb.init(
            project=f"{config['project_name']}_train",
            name=f"epoch_{epoch}",
            config=config)
        for x, y in dloader:
            lr, hr = x.to(device), y.to(device)
            iter_train(lr, hr, model, optim, device, loss_func, wandb)
        if gc:
            gc.collect()
        scheduler["discriminator"].step()
        scheduler["generator"].step()
        if (epoch % config["checkpoint"]) == 0:
            gen = rgb2srgb(predict_one_sample(model["generator"], fake_const, device, tensor2image))
            table = wandb.Tablel(columns=["Generated Tree"])
            wandb.log({"eval_epoch": table.add_data(wandb.Image(gen))}, commit=False)
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
