"""Two-stage custom CNNs: object-proposal segmentation, then crater classification."""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ObjectProposalCNN(nn.Module):
    """Stage 1: segment candidate surface objects; output is one objectness logit map."""
    def __init__(self, in_channels: int = 3, base: int = 24):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat((self.up3(b), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.head(d1)


class CraterClassifierCNN(nn.Module):
    """Stage 2: classify each proposed object crop: 0=other, 1=crater."""
    def __init__(self, in_channels: int = 3, classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32), nn.MaxPool2d(2),
            ConvBlock(32, 64), nn.MaxPool2d(2),
            ConvBlock(64, 128), nn.MaxPool2d(2),
            ConvBlock(128, 192), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(192, classes))

    def forward(self, x):
        return self.classifier(self.features(x))
