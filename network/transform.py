import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchvision import models


class Affine2dPredictor(nn.Module):
    def __init__(self, n_channels=2):
        super(Affine2dPredictor, self).__init__()
        self.first = nn.Conv2d(in_channels=n_channels * 2, out_channels=64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.encoder = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(512, 4)

    def forward(self, in0, in1):
        x = torch.cat([in0, in1], dim=1)
        x = self.first(x)
        x = self.encoder.relu(self.encoder.bn1(x))
        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class Affine2dTransformer(nn.Module):
    """
    Transform a square image.
    It's best to make the 4 ranges centered at (0, 0, 0, 1),
    which means when the input is [0.5, 0.5, 0.5, 0.5], it doesn't change the image at all.
    Important: these ranges should cover the transforms in agent task (see `agent.py`).
    Attributes:
        tx_range (tuple): a range of translation x, relative to the width of image ([-1, 1] for the largest range).
        ty_range (tuple): a range of translation y, relative to the height of image.
        rot_range (tuple): a range of rotation angle by radius, counter-clockwise.
        scale_range (tuple): a range of scale factor, to make this range centered at 1, the min * max should be 1.
        tx_lambda (lambda from tensor to tensor): linear interpolation of range.
        ty_lambda (lambda from tensor to tensor): linear interpolation
        rot_lambda (lambda from tensor to tensor): linear interpolation
        scale_lambda (lambda from tensor to tensor): exp interpolation.
    """
    def __init__(self):
        self.tx_range = (-1.0, 1.0)
        self.ty_range = (-1.0, 1.0)
        self.rot_range = (-math.pi / 3.0, math.pi / 3.0)
        self.scale_range = (0.333333, 3.0)
        self.tx_lambda = lambda x: x * (self.tx_range[1] - self.tx_range[0]) + self.tx_range[0]
        self.ty_lambda = lambda y: y * (self.ty_range[1] - self.ty_range[0]) + self.ty_range[0]
        self.rot_lambda = lambda r: r * (self.rot_range[1] - self.rot_range[0]) + self.rot_range[0]
        self.scale_lambda = lambda s: torch.exp(
            (math.log(self.scale_range[1]) - math.log(self.scale_range[0])) * s + math.log(self.scale_range[0]))
        super(Affine2dTransformer, self).__init__()

    def forward(self, src, params):
        """
        Args:
            src (torch.Tensor): shape of (B, C, H, W)
            params (torch.Tensor): shape of (1, 4), [[tx, ty, rot, scale]], all between [0, 1]
        Returns:
            torch.Tensor: shape as same as src.
        """
        tx, ty, rot, scale = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        tx = self.tx_lambda(tx)
        ty = self.ty_lambda(ty)
        rot = self.rot_lambda(rot)
        scale = self.scale_lambda(scale)
        inv_s = 1.0 / scale
        cos_a = torch.cos(rot)
        sin_a = torch.sin(rot)
        '''
        The inverse of 
            [[s * cos a, -s * sin a, tx],
            [s * sin a, s * cos a, ty],
            [0, 0, 1]].
        '''
        mtx = torch.stack([
            torch.stack([inv_s * cos_a, inv_s * sin_a, -inv_s * (tx * cos_a + ty * sin_a)], dim=1),
            torch.stack([-inv_s * sin_a, inv_s * cos_a, inv_s * (tx * sin_a - ty * cos_a)], dim=1)
        ], dim=1)
        grid = nnf.affine_grid(mtx, src.size(), align_corners=False)
        _out = nnf.grid_sample(src, grid, mode='bilinear', align_corners=False)
        return _out


if __name__ == '__main__':
    a = Affine2dTransformer()
    a(None, torch.Tensor([[0.5, 0.5, 0.5, 0.5]]))
