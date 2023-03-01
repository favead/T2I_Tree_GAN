from typing import Dict, List, Tuple, Union, Callable
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def show_image(img: np.ndarray) -> None:
    plt.imshow(img)
    plt.show()


def visualize_history(data: dict, figsize: Tuple[int, int] = (10, 6)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(x=data["epoch"], y=data["value"], label=data["label"])
    ax.set_ylabel(data["ylabel"])
    ax.set_xlabel(data["xlabel"])
    plt.show()


def visualize_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, data: dict,
                       figsize: Tuple[int, int] = (30, 15), save: bool = False) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_trisurf(np.ravel(X), np.ravel(Y), np.ravel(Z), cmap=plt.cm.viridis, linewidth=0.2)
    ax.set_xlabel(data["xlabel"], fontsize="12",  labelpad=9)
    ax.set_ylabel(data["ylabel"],  fontsize="12", labelpad=10)
    ax.set_zlabel(data["zlabel"], fontsize="22", labelpad=-30)
    ax.view_init(20, 30)
    if save:
        plt.savefig(data["filename"])
    else:
        plt.show()


def create_random_direction(model: nn.Module, weights) -> None:
    direction = [torch.randn(w.size()) for w in weights]
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.fill_(0)
        else:
            for d, w in zip(direction, weights):
                d.mul_(w.norm()/(d.norm() + 1e-10))
    return direction


def generate_data_for_visualize(model: nn.Module, weights: List[Tensor], dloader: DataLoader,
                                xcoords: np.ndarray, ycoords: np.ndarray, loss: Callable,
                                eval: Callable, metric: Callable) -> str:
    d1 = create_random_direction(model, weights)
    d2 = create_random_direction(model, weights)
    plot_data = np.zeros((len(xcoords), len(ycoords), 2))
    for i in range(len(xcoords)):
        for j in range(len(ycoords)):
            changes = [d0_i*xcoords[i] + d1_i*ycoords[j] for (d0_i, d1_i) in zip(d1, d2)]
            for (p, w, d) in zip(model.parameters(), weights, changes):
                p.data = w + torch.Tensor(d).type(type(w))
            loss, metric = eval(model, dloader, loss, metric)
            plot_data[i, j, 0] = loss
            plot_data[i, j, 1] = metric
    return plot_data