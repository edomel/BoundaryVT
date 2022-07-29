from networks.init_weights import init_xavier
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import Conv2d, get_norm


class SmallUpsampleDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 21,
        mid_channels: int = 32,
        num_classes: int = 41,
        ksize=3,
    ):
        super().__init__()

        self.num_classes = num_classes
        norm = "SyncBN"

        self.conv1 = Conv2d(
            in_channels,
            in_channels,
            kernel_size=ksize,
            padding=ksize // 2,
            bias=False,
            norm=get_norm(norm, in_channels),
            activation=F.relu,
        )

        self.conv2 = Conv2d(
            in_channels,
            mid_channels,
            kernel_size=ksize,
            padding=ksize // 2,
            bias=False,
            norm=get_norm(norm, mid_channels),
            activation=F.relu,
        )

        self.conv1.apply(init_xavier)
        self.conv2.apply(init_xavier)

        self.conv3 = Conv2d(mid_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.conv3.weight, 0, 0.001)
        nn.init.constant_(self.conv3.bias, 0)

    def forward(self, in_img: torch.FloatTensor) -> torch.FloatTensor:

        out_img = self.conv1(in_img)

        out_img = F.interpolate(
            out_img, scale_factor=2, mode="bilinear", align_corners=False
        )

        out_img = self.conv2(out_img)

        out_img = self.conv3(out_img)

        return out_img
