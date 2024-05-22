import torch
import torch.nn as nn
from torchvision import models

from utils import timer

class ProFEN(nn.Module):
    def __init__(self, n_channels=2, out_features=2048):
        super(ProFEN, self).__init__()
        self.first = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.encoder = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        self.mlp = nn.Linear(512, out_features)

    @timer
    def forward(self, _in):
        x = self.first(_in)
        x = self.encoder.relu(self.encoder.bn1(x))
        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x
