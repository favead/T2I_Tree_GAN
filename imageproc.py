from typing import Tuple
import math
import numpy as np
import cv2
from torch import Tensor
from torchvision.transforms.functional import to_tensor


def read_image(img_path: str, mode: str = 'RGB') -> np.ndarray:
    img = cv2.imread(img_path)[...,::-1]
    if mode == 'RGB':
        return srgb2rgb(img)
    return img


def image2tensor(img: np.ndarray, normalize: bool = True) -> Tensor:
    img = img / 255.
    tensor = to_tensor(img)
    if normalize:
        tensor = tensor.mul_(2.0).sub_(1.0)
    tensor = tensor.float()
    return tensor


def tensor2image(timg: Tensor, normalize: bool = True, only_y: bool = False) -> np.ndarray:
    ftimg = timg.clone()
    if normalize:
        ftimg.add_(1.0).div_(2.0)
    if only_y:
        img = ftimg.squeeze_(0).mul_(255).clamp_(0, 255).cpu().numpy().astype(np.uint8)
    else:
        img = ftimg.squeeze_(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).cpu().numpy().astype(np.uint8)
    return img


def save_image(fname: str, img: np.ndarray) -> None:
    cv2.imwrite(fname, img[...,::-1])
