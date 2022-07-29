import torch
import torch.nn as nn

from modules.crisp_upsample import CrispUpsample
from modules.hrnet import hrnet32
from modules.decoder import SmallUpsampleDecoder
from detectron2.modeling.backbone.resnet import BottleneckBlock


criterion_dicts = {
    "VT": nn.MSELoss(reduction="none"),
    "DT": nn.L1Loss(reduction="none"),
}


class HRNet32BoundaryNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.boundary_criterion = criterion_dicts[cfg["METHOD_TYPE"]]
        self.boundary_loss_weight = cfg["NETWORK"]["BOUNDARY_LOSS_WEIGHT"]

        self.backbone = hrnet32()
        self.hrnet_out = 256

        self.conv = BottleneckBlock(
            3, 64, bottleneck_channels=32, stride=2, norm="SyncBN"
        )
        self.transfer_conv = BottleneckBlock(
            64, self.hrnet_out // 2, bottleneck_channels=64, num_groups=1, norm="SyncBN"
        )
        self.fuse_layer = CrispUpsample(
            main_channels=self.hrnet_out,
            side_channels=self.hrnet_out // 2,
            out_channels=self.hrnet_out // 4,
        )

        self.boundary_prediction = SmallUpsampleDecoder(
            in_channels=self.hrnet_out // 4,
            mid_channels=32,
            num_classes=cfg["NETWORK"]["OUTPUT_SIZE"],
            ksize=3,
        )

    def forward(self, data_dict):

        # H, W = data_dict["image"].shape[-2:]
        out = {}

        out["main"] = self.backbone(data_dict["image"])

        out["side"] = self.transfer_conv(self.conv(data_dict["image"]))
        out["hid"] = self.fuse_layer(out["main"], out["side"])

        out["boundary_rep"] = self.boundary_prediction(out["hid"])

        if self.training:
            loss = 0

            mask = data_dict["mask"].unsqueeze(1).repeat_interleave(2, dim=1)

            boundary_loss = self.boundary_criterion(
                out["boundary_rep"], data_dict["boundary"]
            )
            loss += (
                torch.sum(boundary_loss * mask) / torch.sum(mask)
            ) * self.boundary_loss_weight

            return loss

        else:

            return out
