"""Short-range sub-branch: DeepLabv3 with a customized ResNet-18 backbone
(Sec. 3.3 / 4.1 of the paper), operating on the downsampled low-frequency
component. The backbone uses a deep stem and dilated stages 3-4 (output
stride 8); context is aggregated with ASPP."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class DilatedResNet18(nn.Module):
    """ResNet-18 with a deep stem and dilated stages 3-4 (output stride 8)."""

    def __init__(self, in_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dilation=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dilation=1)
        self.layer3 = self._make_layer(128, 256, 2, stride=1, dilation=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=1, dilation=4)
        self.out_channels = 512

    @staticmethod
    def _make_layer(in_planes, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(in_planes, planes, stride, dilation, downsample)]
        layers += [BasicBlock(planes, planes, 1, dilation) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.stem(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)                       # (B, 512, H/8, W/8)


class ASPP(nn.Module):
    def __init__(self, in_channels, channels, dilations=(1, 12, 24, 36)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, channels, 1 if d == 1 else 3,
                          padding=0 if d == 1 else d, dilation=d, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            ))
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels, 1, bias=False),
            # GroupNorm: the pooled 1x1 map breaks BatchNorm at batch size 1.
            nn.GroupNorm(1, channels), nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels * (len(dilations) + 1), channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        pooled = F.interpolate(self.image_pool(x), x.shape[2:],
                               mode="bilinear", align_corners=False)
        feats = [branch(x) for branch in self.branches] + [pooled]
        return self.bottleneck(torch.cat(feats, dim=1))


class DeepLabV3(nn.Module):
    def __init__(self, in_channels, channels=128):
        super().__init__()
        self.backbone = DilatedResNet18(in_channels)
        self.aspp = ASPP(self.backbone.out_channels, channels)
        self.out_channels = channels

    def forward(self, x):
        return self.aspp(self.backbone(x))          # (B, C, H/8, W/8)
