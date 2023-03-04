from typing import List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models import VGG19_BN_Weights, vgg19_bn
from torch import nn, Tensor
from config import Config


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

    
class PixelShuffleBlock(nn.Module):
    def __init__(self, factor: int, in_c: int, out_c: int) -> None:
        super(PixelShuffleBlock, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module("Conv", nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=(1, 1)))
        self.block.add_module("PixelShuffle", nn.PixelShuffle(factor))
        self.block.add_module("PReLU", nn.PReLU())
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return out
    

class Generator(nn.Module):
    def __init__(self, img_ch: int = 3, conv_ch: int = 64, factor: int = 2) -> None:
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(img_ch, conv_ch, kernel_size=(9, 9), padding=(9//2, 9//2))
        self.prelu1 = nn.PReLU()
        grbs = []
        for _ in range(16):
            grbs.append(GenResBlock(conv_ch, conv_ch))
        self.grbs = nn.Sequential(*grbs)
        self.conv2 = nn.Conv2d(conv_ch, conv_ch, kernel_size=(3, 3), padding=(1, 1))
        self.pxlshfl = PixelShuffleBlock(factor, conv_ch, int(conv_ch*(factor ** 2.)))
        self.conv3 = nn.Conv2d(conv_ch, img_ch, kernel_size=(9, 9), padding=(9//2, 9//2))
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out1 = self.prelu1(self.conv1(x))
        out2 = self.grbs(out1)
        out3 = self.conv2(out2)
        out3 = torch.add(out3, out1)
        out4 = self.pxlshfl(out3)
        out5 = self.conv3(out4)
        out6 = torch.tanh(out5)
        return out6
    

class DisBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: Tuple[int, int]) -> None:
        super(DisBlock, self).__init__()
        self.block = nn.Sequential()
        self.block.add_module("Conv", nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=stride))
        self.block.add_module("BatchNorm", nn.BatchNorm2d(out_c))
        self.block.add_module("LRelu", nn.LeakyReLU(negative_slope=0.2))
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return out
    

class Discriminator(nn.Module):
    def __init__(self, img_c: int = 3, conv_ch: int = 64, conv_s: Tuple[int, int] = (2, 2)) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(img_c, conv_ch, kernel_size=(3, 3))
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)
        dis_blocks = []
        strides = [conv_s, (1, 1)]
        for i in range(7):
            dis_blocks.append(DisBlock(conv_ch * int(2 ** (i // 2)),
                                       conv_ch * int(2 ** ((i + 1) // 2)),
                                       strides[int(i%2)]))
        self.discs = nn.Sequential(*dis_blocks)
        self.linear1 = nn.Linear(4608, 1024)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2)
        self.linear2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
        return None
    
    def forward(self, x: Tensor) -> Tensor:
        out1 = self.lrelu1(self.conv1(x))
        out2 = self.discs(out1)
        out3 = self.lrelu2(self.linear1(torch.flatten(out2, 1)))
        out4 = self.sigmoid(self.linear2(out3))
        return out4


def get_vgg_model(device: torch.device) -> nn.Module:
    vgg = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    modules = list(vgg.features.requires_grad_(False).children())[:Config.vgg_layers[-1] + 1]
    return modules.to(device)


def get_vgg_maps(vgg_modules: nn.Module, fake: Tensor, real: Tensor, device: torch.device) -> List[Tensor]:
    vgg_loss = []
    x = (fake.clone() + 1.) / 2.
    y = (real.clone() + 1.) / 2.
    for i in range(Config.vgg_layers[-1]):
        module = vgg_modules[i]
        x = module(x)
        y = module(y)
        if i in Config.vgg_layers:
            vgg_loss.append(F.l1_loss(x, y))
    return vgg_loss


def adversarial_loss(preds: Tensor, gt: Tensor) -> Tensor:
    loss = F.mse_loss(preds, gt)
    return loss


def perceptual_loss(fake: Tensor, real: Tensor, preds: Tensor, gt: Tensor,
                     vgg_modules: List[nn.Module]) -> Tuple[Tensor, Tensor, Tensor]:
    vgg_loss = sum(get_vgg_maps(vgg_modules, fake, real))
    pixel_loss = F.l1_loss(real, fake)
    adv_loss = F.mse_loss(preds, gt)
    return vgg_loss, pixel_loss, adv_loss


def save_weights(model: nn.Module, weight_dir: str) -> None:
    torch.save(model, weight_dir)