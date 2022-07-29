import torch
import yaml

from modules.options import parse_arguments

import torch.multiprocessing

from modules.trainer import Trainer
from modules.validation import validate

torch.multiprocessing.set_sharing_strategy("file_system")


def main():

    args = parse_arguments()
    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda:{}".format(cfg["DEVICE_ID"]))

    trainer = Trainer(cfg, args, device)

    if args.training:
        print("Executing training")

        trainer.train()

    else:
        print("Executing validation")

        validate(
            cfg, trainer.net, trainer.validation_set, trainer.validation_loader, device
        )

    return


if __name__ == "__main__":
    main()
