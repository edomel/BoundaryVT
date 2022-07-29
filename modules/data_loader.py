import torch

from torchvision.transforms import Normalize
from detectron2.data import transforms as T

import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from os import walk

PI = (torch.abs(torch.acos(torch.zeros(1))).item()) * 2
MAX_VALUE = 255 * 256 ** 2 + 255 * 256 + 255


def convert_representation(img, min=-PI, max=PI):

    restored_img = torch.zeros_like(img[0])
    restored_img = img[0] + img[1] * 256 + img[2] * 256 ** 2
    restored_img = (restored_img * (max - min) / MAX_VALUE) + min

    return restored_img


def restore_representation(img, min=-PI, max=PI):
    img_processing = img.clone()
    img_processing -= min
    max_range = max - min
    img_processing = (img_processing * MAX_VALUE / max_range).type(torch.LongTensor)
    img3 = img_processing // (256 ** 2)
    img2 = (img_processing % (256 ** 2)) // 256
    img1 = (img_processing % (256 ** 2)) % 256
    encoded_img = torch.stack((img1, img2, img3), dim=-1)

    return encoded_img


class BoundaryImageSet(Dataset):
    def __init__(
        self,
        data_dir,
        dataset_structure,
        phase="training",
        augmentations=None,
        give_field=False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.give_field = give_field
        self.phase = phase
        self.dataset_structure = dataset_structure

        if self.phase == "training":
            self.augmentations = augmentations
            self.augmentations = T.AugmentationList(augmentations)

        self.normalize = Normalize(
            mean=[123.675, 116.280, 103.530], std=[58.395, 57.120, 57.375]
        )

        idx = 0
        self.image_set = {}
        for (dirpath, dirnames, filenames) in walk(self.data_dir):
            if (self.phase in dirpath) and (self.dataset_structure["IMAGE"] in dirpath):
                for filename in filenames:

                    data_id = idx
                    self.image_set[data_id] = {}
                    self.image_set[data_id]["image"] = Path(dirpath) / Path(filename)

                    instance_path = str(dirpath).replace(
                        self.dataset_structure["IMAGE"],
                        self.dataset_structure["PANOPTIC"],
                    )
                    instance_file = str(filename).replace(
                        self.dataset_structure["IMAGE_EXT"],
                        self.dataset_structure["PANOPTIC_EXT"],
                    )
                    self.image_set[data_id]["instance"] = Path(instance_path) / Path(
                        instance_file
                    )

                    angle_path = str(dirpath).replace(
                        self.dataset_structure["IMAGE"], self.dataset_structure["ANGLE"]
                    )
                    angle_file = str(filename).replace(
                        self.dataset_structure["IMAGE_EXT"],
                        self.dataset_structure["ANGLE_EXT"],
                    )
                    self.image_set[idx]["angle"] = Path(angle_path) / Path(angle_file)

                    dt_path = str(dirpath).replace(
                        self.dataset_structure["IMAGE"], self.dataset_structure["DT"]
                    )
                    dt_file = str(filename).replace(
                        self.dataset_structure["IMAGE_EXT"],
                        self.dataset_structure["DT_EXT"],
                    )
                    self.image_set[idx]["dt"] = Path(dt_path) / Path(dt_file)

                    idx += 1

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):

        return self.process_data(self.image_set[idx], idx)

    def process_data(self, data, img_id):

        sample = {}
        sample["img_id"] = img_id

        img = (
            torch.from_numpy(np.array(Image.open(data["image"])))
            .type(torch.FloatTensor)
            .permute(2, 0, 1)
        )
        if img.amax() < 10:
            img *= 255.0

        instance = (
            torch.from_numpy(np.array(Image.open(data["instance"])))
            .type(torch.FloatTensor)
            .permute(2, 0, 1)
        )
        instance = instance[0]
        # H, W = instance.shape

        angle = (
            torch.from_numpy(np.array(Image.open(data["angle"])))
            .type(torch.FloatTensor)
            .permute(2, 0, 1)
        )
        angle = convert_representation(angle)

        dt = (
            torch.from_numpy(np.array(Image.open(data["dt"])))
            .type(torch.FloatTensor)
            .permute(2, 0, 1)
        )
        dt = convert_representation(dt, min=0, max=4000.0)

        img = img.permute(1, 2, 0).numpy().astype(float)
        instance = instance.numpy().astype(float)
        angle = angle.numpy().astype(float)
        dt = dt.numpy().astype(float)

        if self.phase == "training":
            aug_input = T.AugInput(img)
            tfms = self.augmentations(aug_input)
            img = aug_input.image

            size_rescaling_parameter = tfms[0].new_h / tfms[0].h

            for i in range(len(tfms)):

                if isinstance(tfms[i], T.ResizeTransform):
                    instance = tfms[i].apply_image(instance, interp=Image.NEAREST)
                    angle = tfms[i].apply_image(angle, interp=Image.NEAREST)
                    dt = tfms[i].apply_image(dt, interp=Image.NEAREST)
                else:
                    instance = tfms[i].apply_image(instance)
                    angle = tfms[i].apply_image(angle)
                    dt = tfms[i].apply_image(dt)
        else:
            # During validation or testing no rescaling is applied
            size_rescaling_parameter = 1.0

        # Rescale angles (just for easy representation understanding)
        angle_image_to_return = torch.from_numpy(angle.copy()).float()
        angle_image_to_return = (angle_image_to_return * 90.0) / (PI / 2.0)

        # Adapt dt to account for rescaling augmentations
        dt_image_to_return = torch.from_numpy(dt.copy()).float()
        dt_image_to_return *= size_rescaling_parameter

        img = torch.from_numpy(img.copy()).float().permute(2, 0, 1)
        instance = torch.from_numpy(instance.copy()).float()

        if self.phase == "training":

            if isinstance(tfms[-1], T.HFlipTransform):
                # Adapt angle if image has been flipped
                angle_image_to_return = 180 - angle_image_to_return
                angle_image_to_return[angle_image_to_return > 180] -= 360

        sample["angle"] = angle_image_to_return
        sample["dt"] = dt_image_to_return
        sample["image"] = self.normalize(img)
        sample["instance"] = self.instance_image_creation(instance)

        boundary_mask = torch.zeros_like(sample["instance"])
        boundary_mask[:-1] += torch.abs(
            sample["instance"][:-1] - sample["instance"][1:]
        )
        boundary_mask[:, :-1] += torch.abs(
            sample["instance"][:, :-1] - sample["instance"][:, 1:]
        )
        boundary_mask[1:] += boundary_mask[:-1].clone()
        boundary_mask[:, 1:] += boundary_mask[:, :-1].clone()
        boundary_mask[boundary_mask > 1] = 1
        sample["boundary_mask"] = boundary_mask

        if self.give_field:

            (
                sample["field"],
                sample["field_mask"],
            ) = self.get_continuous_target_from_angle(sample["angle"].clone())

        return sample

    def instance_image_creation(self, label_img):

        _, inverse_indices = torch.unique(label_img, sorted=True, return_inverse=True)

        return inverse_indices

    def get_continuous_target_from_angle(self, angle_img):

        # If angle_img has values lower than -500 those should not be trusted
        # Not used in practice but can be used to exclude pixels with unreliable angle values
        mask = angle_img.clone()
        angle_field = torch.stack(
            (
                torch.sin(angle_img * (PI / 2.0) / 90.0),
                torch.cos(angle_img * (PI / 2.0) / 90.0),
            ),
            dim=0,
        )
        angle_field[:, mask < -500] = 0
        mask[mask > -500] = 1
        mask[mask <= -500] = 0

        return angle_field, mask
