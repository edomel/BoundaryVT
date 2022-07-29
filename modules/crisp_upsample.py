import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import Conv2d, get_norm

from networks.init_weights import init_xavier


class CrispUpsample(nn.Module):
    def __init__(self, main_channels, side_channels, out_channels):
        super().__init__()

        self.conv_main = Conv2d(
            main_channels,
            main_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv_side = Conv2d(
            side_channels,
            side_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.fuse_conv = Conv2d(
            side_channels // 2 + main_channels // 2,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=get_norm("SyncBN", out_channels),
            activation=F.relu,
        )
        self.conv_main.apply(init_xavier)
        self.conv_side.apply(init_xavier)
        self.fuse_conv.apply(init_xavier)

    def forward(self, main_img, side_img):

        main_img = F.interpolate(
            main_img, size=side_img.size()[2:], mode="bilinear", align_corners=False
        )

        img = torch.cat((self.conv_main(main_img), self.conv_side(side_img)), dim=1)
        img = self.fuse_conv(img)

        return img
