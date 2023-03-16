from typing import List, Tuple, Union, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch import nn, Tensor


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if (classname.find('Conv') | classname.find('ConvTranspose')) != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    return None


class GenTransposeBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int, padding: int) -> None:
        super(GenTransposeBlock, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=stride,
                                                                padding=padding)
        self.prelu = nn.PReLU()
        self.tconv2 = nn.ConvTranspose2d(out_c, out_c, kernel_size=1, stride=1,
                                                                padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.tconv1(x)
        out2 = self.prelu(out1)
        out3 = self.tconv2(out2)
        out4 = torch.add(out1, out3)
        out4 = self.prelu(out4)
        return out4


class Generator(nn.Module):
    def __init__(self, inp_dim: int) -> None:
        super(Generator, self).__init__()
        # init: (BS, 100, 1, 1)
        self.tconv1 = GenTransposeBlock(inp_dim, 1024, 1, 1) # (BS, 1024, 2, 2)
        self.tconv2 = GenTransposeBlock(1024, 512, 2, 1) # (BS, 512, 4, 4)
        self.tconv3 = GenTransposeBlock(512, 128, 2, 1) # (BS, 128, 8, 8)
        self.tconv4 = GenTransposeBlock(128, 3, 2, 1) # (BS, 3, 16, 16)
    
    def forward(self, x: Tensor) -> Tensor:
        out1 = self.tconv1(x)
        out2 = self.tconv2(out1)
        out3 = self.tconv3(out2)
        out4 = self.tconv4(out3)
        out5 = torch.tanh(out4)
        return out5


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
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return out
    

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__() # 16 16 3
        self.conv1 = DiscBlock(3, 128, 2, 1) # 3 8 8
        self.conv2 = DiscBlock(128, 256, 2, 1) # 256 4 4
        self.conv3 = DiscBlock(304, 512, 2, 1) # 512 2 2
        self.conv4 = DiscBlock(512, 1, 2, 1, is_last=True) # 1 1 1

    def forward(self, x: Tensor, embed: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out2 = torch.hstack((out2, embed.view(embed.size()[0], 48, 4, 4)))
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out4 = torch.flatten(out4, 1)
        return out4


def discriminator_loss(sr: Tensor, sw: Tensor, sf: Tensor, device: torch.device) -> Tensor:
    sr_l = torch.ones((len(sr), 1)).to(device)
    sw_sf_l = torch.zeros((len(sw), 1)).to(device)
    loss = F.binary_cross_entropy(sr, sr_l)
    loss += 0.5 * (F.binary_cross_entropy(sw, sw_sf_l) + F.binary_cross_entropy(sf, sw_sf_l))
    return loss


def generator_loss(sf: Tensor, device: torch.device) -> Tensor:
    sf_l = torch.ones((len(sf), 1)).to(device)
    loss = F.binary_cross_entropy(sf, sf_l)
    return loss


def save_weights(model: nn.Module, weight_dir: str) -> None:
    torch.save(model, weight_dir)