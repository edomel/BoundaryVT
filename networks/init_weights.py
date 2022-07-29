import torch
import torch.nn as nn
import torch.nn.init as I


def init_kaiming(layer):

    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        I.kaiming_normal_(layer.weight, nonlinearity="relu", mode="fan_out")
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)

    return


def init_xavier(layer):

    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        I.kaiming_uniform_(layer.weight, a=1)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)

    return


def init_normal(layer):

    if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        I.normal_(layer.weight, std=0.001)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)

    return
