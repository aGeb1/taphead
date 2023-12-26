import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=False):
        super().__init__()

        if stride:
            self.identity = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=1
            )
        else:
            self.identity = nn.Identity()
            stride = 1

        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv0 = self.identity = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        y = self.conv0(nn.ReLU(self.bn0(x)))
        y = self.conv1(nn.ReLU(self.bn1(x)))
        return y + self.identity(x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, N, stride):
        super().__init__()
        self.conv0 = Residual(in_channels, out_channels, stride=stride)
        self.convn = []
        for _ in range(1, N):
            self.convn.append(Residual(out_channels, out_channels))


class ResNet(nn.Module):
    def __init__(self, k, N):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1 = Block(32, 16, 16*k, N, 1)
        self.conv2 = Block(16, 16*k, 32*k, N, 2)
        self.conv3 = Block(8, 32*k, 64*k, N, 2)
        self.linear = nn.Linear(64*k, 10)

        self.bn = nn.BatchNorm2d(64*k)
        self.pool = nn.AvgPool2d(8, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.bn(x)
        x = nn.ReLU(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.linear(x)
        return x

