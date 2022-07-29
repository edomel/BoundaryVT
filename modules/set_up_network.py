import torch
from networks.HRNet32_boundaries import HRNet32BoundaryNet

from modules.warmup_poly_lr import WarmupPolyLR
from modules.revert_sync_bn import revert_sync_batchnorm
from pathlib import Path


network_dicts = {
    "hrnet32_boundary_net": HRNet32BoundaryNet,
}


def set_up_network(cfg, args, device):

    net = network_dicts[cfg["NETWORK"]["ARCHITECTURE"]](cfg, args)
    net = revert_sync_batchnorm(net)
    learning_rate = cfg["NETWORK"]["LEARNING_RATE"]
    net_optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
    num_iterations = cfg["NETWORK"]["NUM_ITERATIONS"]

    if cfg["NETWORK"]["SCHEDULER"] == "WarmupPolyLR":
        net_sched = WarmupPolyLR(net_optim, num_iterations)
    else:
        net_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            net_optim, num_iterations
        )

    iteration = 0
    if cfg["NETWORK"]["LOAD_CHECKPOINT"]:
        print("Loading saved checkpoint", cfg["NETWORK"]["CHECKPOINT"])
        checkpoint = torch.load(
            Path(cfg["DATA_DIR"])
            / Path("../" + cfg["METHOD_NAME"])
            / Path("checkpoint_{}.pt".format(cfg["NETWORK"]["CHECKPOINT"])),
            map_location=device,
        )
        net.load_state_dict(checkpoint["model_state_dict"])
        net_optim.load_state_dict(checkpoint["optimizer_state_dict"])
        net_sched.load_state_dict(checkpoint["scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        for state in net_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    net = net.to(device)

    return net, net_optim, net_sched, iteration
