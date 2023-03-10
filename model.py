from typing import List, Tuple, Union, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch import nn, Tensor


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    return None


class GenTransposeBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int, padding: int,
                  is_last: bool = False) -> None:
        super(GenTransposeBlock, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module("ConvTranspose", nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=stride,
                                                                   padding=padding))
        if is_last:
            self.block.add_module("Tanh", nn.Tanh())
        else:
            self.block.add_module("BatchNorm", nn.BatchNorm2d(out_c))
            self.block.add_module("ReLU", nn.ReLU(inplace=True))
        return None

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # init: (BS, 100, 1, 1)
        self.tconv1 = GenTransposeBlock(100, 1024, 1, 1) # (BS, 1024, 2, 2)
        self.tconv2 = GenTransposeBlock(1024, 512, 2, 1) # (BS, 512, 4, 4)
        self.tconv3 = GenTransposeBlock(512, 128, 2, 1) # (BS, 128, 8, 8)
        self.tconv4 = GenTransposeBlock(128, 3, 2, 1, is_last=True) # (BS, 3, 16, 16)
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out1 = self.tconv1(x)
        out2 = self.tconv2(out1)
        out3 = self.tconv3(out2)
        out4 = self.tconv4(out3)
        return out4


class DiscBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int, padding: int,
                 is_last: bool = False) -> None:
        super(DiscBlock, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module("Conv", nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride,
                                                padding=padding))
        if is_last:
            self.block.add_module("Sigmoid", nn.Sigmoid())
        else:
            self.block.add_module("BatchNorm", nn.BatchNorm2d(out_c))
            self.block.add_module("LRelu", nn.LeakyReLU(negative_slope=0.2))
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return out
    

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__() # 16 16 3
        self.conv1 = DiscBlock(3, 128, 2, 1)
        self.conv2 = DiscBlock(128, 256, 2, 1)
        self.conv3 = DiscBlock(256, 512, 2, 1)
        self.conv4 = DiscBlock(512, 1, 2, 1, is_last=True)
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out4 = torch.flatten(out4, 1)
        return out4


def adversarial_loss(preds: Tensor, gt: Tensor) -> Tensor:
    loss = F.binary_cross_entropy(preds, gt)
    return loss


def save_weights(model: nn.Module, weight_dir: str) -> None:
    torch.save(model, weight_dir)