from typing import List, Tuple, Union, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch import nn, Tensor


class GenResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super(GenResBlock, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module("Conv1", nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=(1, 1)))
        self.block.add_module("PReLU", nn.PReLU())
        self.block.add_module("Conv2", nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=(1, 1)))
        return None

    def forward(self, x: Tensor) -> Tensor:
        identity = x.clone()
        out = self.block(x)
        return torch.add(identity, out)


class Generator(nn.Module):
    def __init__(self, img_ch: int = 3, conv_ch: int = 64) -> None:
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(img_ch, conv_ch, kernel_size=(9, 9), padding=(9//2, 9//2))
        self.prelu1 = nn.PReLU()
        grbs = []
        for _ in range(8):
            grbs.append(GenResBlock(conv_ch, conv_ch))
        self.grbs = nn.Sequential(*grbs)
        self.conv2 = nn.Conv2d(conv_ch, conv_ch, kernel_size=(3, 3), padding=(1, 1))
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(conv_ch, img_ch, kernel_size=(9, 9), padding=(9//2, 9//2))
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out1 = self.prelu1(self.conv1(x))
        out2 = self.grbs(out1)
        out3 = self.prelu2(self.conv2(out2))
        out3 = torch.add(out3, out1)
        out4 = torch.tanh(self.conv3(out3))
        return out4
    

class DisBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: Tuple[int, int], kernel_size=(3, 3)) -> None:
        super(DisBlock, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module("Conv", nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride))
        self.block.add_module("BatchNorm", nn.BatchNorm2d(out_c))
        self.block.add_module("LRelu", nn.LeakyReLU(negative_slope=0.2))
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return out
    

class Discriminator(nn.Module):
    def __init__(self, img_c: int = 3, conv_ch: int = 64, conv_s: Tuple[int, int] = (2, 2)) -> None:
        super(Discriminator, self).__init__() # 16 16 3
        self.conv1 = nn.Conv2d(img_c, conv_ch, kernel_size=(3, 3)) # 14, 14
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)
        self.disc_block1 = DisBlock(conv_ch, conv_ch * 2, (2, 2)) # 6, 6
        self.disc_block2 = DisBlock(conv_ch * 2, conv_ch * 4, (1, 1)) # 4, 4
        self.disc_block3 = DisBlock(conv_ch * 4, conv_ch * 8, (1, 1)) # 2, 2
        self.disc_block4 = DisBlock(conv_ch * 8, conv_ch * 16, (1, 1), kernel_size=(2, 2)) # 1, 1
        self.linear1 = nn.Linear(conv_ch * 16, 1024)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2)
        self.linear2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out1 = self.lrelu1(self.conv1(x))
        out2 = self.disc_block1(out1)
        out3 = self.disc_block2(out2)
        out4 = self.disc_block3(out3)
        out5 = self.disc_block4(out4)
        out6 = self.lrelu2(self.linear1(torch.flatten(out5, 1)))
        out7 = self.sigmoid(self.linear2(out6))
        return out7


def adversarial_loss(preds: Tensor, gt: Tensor) -> Tensor:
    loss = F.mse_loss(preds, gt)
    return loss


def save_weights(model: nn.Module, weight_dir: str) -> None:
    torch.save(model, weight_dir)