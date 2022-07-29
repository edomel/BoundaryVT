import torch
import time

from torch.utils.data import DataLoader
from pathlib import Path
from modules.augmentations import build_sem_seg_train_aug

from modules.data_loader import BoundaryImageSet
from modules.method_data_prep import get_method_specific_data_dict
from modules.set_up_network import set_up_network
from modules.validation import validate


class Trainer:
    def __init__(self, cfg, args, device) -> None:
        self.cfg = cfg
        self.args = args
        self.device = device
        self.batch = self.args.batch
        augmentations = build_sem_seg_train_aug(cfg["AUGMENTATIONS"])

        self.train_set = BoundaryImageSet(
            cfg["DATA_DIR"],
            cfg["DATASET_STRUCTURE"],
            phase="training",
            augmentations=augmentations,
            give_field=True,
        )
        self.training_loader = DataLoader(
            self.train_set,
            batch_size=self.batch,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.cfg["NUM_WORKERS"],
        )

        self.validation_set = BoundaryImageSet(
            cfg["DATA_DIR"],
            cfg["DATASET_STRUCTURE"],
            phase="validation",
            give_field=True,
        )
        self.validation_loader = DataLoader(
            self.validation_set,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=1,
        )

        self.net, self.net_optim, self.net_sched, iteration = set_up_network(
            self.cfg, self.args, self.device
        )
        num_iterations = cfg["NETWORK"]["NUM_ITERATIONS"]
        self.num_epochs = num_iterations * self.args.batch // len(self.train_set)
        self.cur_epoch = (iteration * self.args.batch) // len(self.train_set)
        pass

    def train(self):

        self.net.train()
        self.net.zero_grad()

        best_score = 0.0
        best_epoch = 0

        for epoch in range(self.cur_epoch, self.num_epochs):
            start = time.time()

            averaged_loss = 0.0

            for j, data_point in enumerate(self.training_loader):

                data_dict = get_method_specific_data_dict(
                    self.cfg["METHOD_TYPE"], data_point, self.device
                )

                complete_loss = self.net(data_dict)

                averaged_loss += complete_loss.item()

                complete_loss.backward()
                self.net_optim.step()
                self.net_optim.zero_grad()
                self.net_sched.step()

            stop = time.time()

            if epoch % 10 == 0 and epoch > 0:

                print(
                    "Averaged loss {}\nEpoch {} of {}\nExecution time {}".format(
                        averaged_loss / len(self.train_set) * self.args.batch,
                        epoch,
                        self.num_epochs,
                        (stop - start) / 60.0,
                    )
                )

                if self.cfg["VALIDATE"]:
                    score = validate(
                        self.cfg,
                        self.net,
                        self.validation_set,
                        self.validation_loader,
                        self.device,
                    )
                    if score > best_score:
                        best_score = score
                        best_epoch = epoch
                    self.net.train()
                    self.net.zero_grad()

                torch.save(
                    {
                        "iteration": epoch * len(self.train_set) // 32,
                        "model_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": self.net_optim.state_dict(),
                        "scheduler_state_dict": self.net_sched.state_dict(),
                    },
                    Path(self.cfg["DATA_DIR"])
                    / Path("../" + self.cfg["METHOD_NAME"])
                    / Path("checkpoint_{}.pt".format(epoch)),
                )

        print(
            "Best boundary assd accuracy {} at epoch {}".format(best_score, best_epoch)
        )

        return
