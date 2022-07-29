import torch
import faiss

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


class FieldExtractionImageSet(Dataset):
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

        if self.dataset_structure["DATASET_NAME"] == "mapillary":
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

        instance = (
            torch.from_numpy(np.array(Image.open(data["instance"])))
            .type(torch.FloatTensor)
            .permute(2, 0, 1)
        )
        instance = instance[0]
        # H, W = instance.shape

        img = img.permute(1, 2, 0).numpy().astype(float)
        instance = instance.numpy().astype(float)

        if self.dataset_structure["DATASET_NAME"] == "mapillary":
            aug_input = T.AugInput(img)
            tfms = self.augmentations(aug_input)
            img = aug_input.image

            for i in range(len(tfms)):

                if isinstance(tfms[i], T.ResizeTransform):
                    instance = tfms[i].apply_image(instance, interp=Image.NEAREST)
                else:
                    instance = tfms[i].apply_image(instance)

        img = torch.from_numpy(img.copy()).float().permute(2, 0, 1)
        instance = torch.from_numpy(instance.copy()).float()

        angle_image, dt_image = self.extract_field(instance)

        sample["angle"] = angle_image
        sample["dt"] = dt_image
        sample["image"] = img
        sample["instance"] = instance

        sample["angle_file_name"] = data["angle"]
        sample["dt_file_name"] = data["dt"]
        sample["image_file_name"] = data["image"]
        sample["instance_file_name"] = data["instance"]

        return sample

    def extract_field(self, label_img, edge_mask=None, num_points=7):

        H, W = label_img.shape

        image_grid = torch.stack(
            (
                torch.arange(H).repeat(W, 1).transpose(0, 1),
                torch.arange(W).repeat(H, 1),
            ),
            dim=0,
        )

        if edge_mask is None:
            edge_mask = torch.zeros((H, W)).type(torch.FloatTensor)
            edge_mask[:-1] += torch.abs(label_img[:-1] - label_img[1:])
            edge_mask[:, :-1] += torch.abs(label_img[:, :-1] - label_img[:, 1:])
            edge_mask[1:] += edge_mask[:-1].clone()
            edge_mask[:, 1:] += edge_mask[:, :-1].clone()
            edge_mask[edge_mask > 1] = 1

        edge_points = np.array(
            image_grid[:, edge_mask == 1].type(torch.FloatTensor).transpose(0, 1)
        ).astype("float32")

        non_edge_points = np.array(
            image_grid[:, edge_mask == 0].type(torch.FloatTensor).transpose(0, 1)
        ).astype("float32")

        edge_points = edge_points.copy(order="C")
        non_edge_points = non_edge_points.copy(order="C")

        index = faiss.IndexFlatL2(2)
        index.add(edge_points)
        D, I = index.search(non_edge_points, num_points)

        closest_point = torch.mean(torch.from_numpy(edge_points[I, :]), dim=1)

        index = faiss.IndexFlatL2(2)
        index.add(non_edge_points)
        D, I = index.search(edge_points, 1)

        non_edge_points = torch.from_numpy(non_edge_points)
        edge_points = torch.from_numpy(edge_points)

        non_edge_angle = torch.atan2(
            -(closest_point[:, 0] - non_edge_points[:, 0]),
            (closest_point[:, 1] - non_edge_points[:, 1]),
        )

        # l2 distance between the closest point and the non edge point, expects row vectors with batches
        non_edge_dt = torch.linalg.norm(closest_point - non_edge_points, dim=1)

        angle_map = torch.zeros((H, W)).type(torch.FloatTensor)
        angle_map[edge_mask == 1] = non_edge_angle[I[:, 0]]
        angle_map[edge_mask == 0] = non_edge_angle

        dt_map = torch.zeros((H, W)).type(torch.FloatTensor)
        dt_map[edge_mask == 1] = 0
        dt_map[edge_mask == 0] = non_edge_dt
        dt_map[edge_mask == -1] = -1

        return angle_map, dt_map
