from torch import nn
import torch
import torch.nn.functional as nnf
from torchvision import models

from .common import UpConv, DoubleConv


class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        resnet = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        chs = [64, 128, 256, 512]
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4
        self.dec4 = UpConv(chs[3], chs[2])
        self.dec3 = UpConv(chs[2], chs[1])
        self.dec2 = UpConv(chs[1], chs[0])
        self.dec1 = DoubleConv(chs[0] + chs[0],  chs[0])
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        return

    def forward(self, last_image, new_image, last_label):
        """
        Estimate the transform between last_image and new_image,
        apply it on last_label to generate new_label.
        Args:
            last_image (torch.Tensor): size(B, C=3, H, W)
            new_image (torch.Tensor): size(B, C=3, H, W)
            last_label (torch.Tensor): size(B, C'=2, H, W)
        Returns:
            new_label (torch.Tensor): size(B, C', H, W) 
            grid (torch.Tensor): size(B, H, W, 2)
        """
        x = torch.cat([last_image, new_image, last_label], dim=1)
        f0 = self.first(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        x = self.dec4(f4, f3)
        x = self.dec3(x, f2)
        x = self.dec2(x, f1)
        x = self.dec1(torch.cat([f0, x], dim=1))
        vector = self.last(x).permute(0, 2, 3, 1)
        bs = last_label.size(0)
        sz = last_label.size()
        base = nnf.affine_grid(torch.eye(3)[:2].repeat(bs, 1, 1), sz).to(vector.device)
        grid = vector + base
        new_label = nnf.grid_sample(last_label, grid, mode='nearest')
        return new_label, vector
