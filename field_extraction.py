import yaml
import torch
import argparse

from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from modules.field_extraction_loader import FieldExtractionImageSet

import detectron2.data.transforms as T
import numpy as np

PI = (torch.acos(torch.zeros(1)).item()) * 2
MAX_VALUE = 255 * 256 ** 2 + 255 * 256 + 255


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="./configs/cityscapes_field_extraction.yaml",
        type=str,
        help="Config file location",
    )
    args = parser.parse_args()
    return args


def convert_representation(img, min=-PI, max=PI):
    img_processing = img.clone()
    img_processing -= min
    max_range = max - min
    img_processing = (img_processing * MAX_VALUE / max_range).type(torch.LongTensor)
    img3 = img_processing // (256 ** 2)
    img2 = (img_processing % (256 ** 2)) // 256
    img1 = (img_processing % (256 ** 2)) % 256
    encoded_img = torch.stack((img1, img2, img3), dim=-1)

    return encoded_img


def build_sem_seg_train_aug():
    augs = [T.ResizeShortestEdge(1024, 4000, "choice")]
    return augs


def main():

    args = parse_arguments()
    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)

    data_set = FieldExtractionImageSet(
        cfg["DATA_DIR"],
        cfg["DATASET_STRUCTURE"],
        phase="training",
        augmentations=build_sem_seg_train_aug(),
        give_field=True,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=cfg["NUM_WORKERS"],
    )

    for j, data_point in enumerate(data_loader):
        if j % 100 == 0:
            print("Training set {} of {}".format(j, len(data_set)))

        img = data_point["image"][0]
        instance = data_point["instance"][0]
        angle = data_point["angle"][0]
        dt = data_point["dt"][0]

        encoded_instance = torch.stack(
            (instance, torch.zeros_like(instance), torch.zeros_like(instance)), dim=-1
        )
        encoded_angle = np.array(convert_representation(angle)).astype(np.uint8)
        encoded_dt = np.array(convert_representation(dt, min=0, max=4000)).astype(
            np.uint8
        )

        img_im = Image.fromarray(np.array(img).astype(np.uint8))
        instance_im = Image.fromarray(np.array(encoded_instance).astype(np.uint8))
        angle_im = Image.fromarray(encoded_angle)
        dt_im = Image.fromarray(encoded_dt)

        img_im.save(data_point["image_file_name"][0])
        instance_im.save(data_point["instance_file_name"][0])
        angle_im.save(data_point["angle_file_name"][0])
        dt_im.save(data_point["dt_file_name"][0])

    data_set = FieldExtractionImageSet(
        cfg["DATA_DIR"],
        cfg["DATASET_STRUCTURE"],
        phase="validation",
        augmentations=build_sem_seg_train_aug(),
        give_field=True,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=cfg["NUM_WORKERS"],
    )

    for j, data_point in enumerate(data_loader):
        if j % 100 == 0:
            print("Training set {} of {}".format(j, len(data_set)))

        img = data_point["image"][0]
        instance = data_point["instance"][0]
        angle = data_point["angle"][0]
        dt = data_point["dt"][0]

        encoded_instance = torch.stack(
            (instance, torch.zeros_like(instance), torch.zeros_like(instance)), dim=-1
        )
        encoded_angle = np.array(convert_representation(angle)).astype(np.uint8)
        encoded_dt = np.array(convert_representation(dt, min=0, max=4000)).astype(
            np.uint8
        )

        img_im = Image.fromarray(np.array(img).astype(np.uint8))
        instance_im = Image.fromarray(np.array(encoded_instance).astype(np.uint8))
        angle_im = Image.fromarray(encoded_angle)
        dt_im = Image.fromarray(encoded_dt)

        img_im.save(data_point["image_file_name"][0])
        instance_im.save(data_point["instance_file_name"][0])
        angle_im.save(data_point["angle_file_name"][0])
        dt_im.save(data_point["dt_file_name"][0])

    data_set = FieldExtractionImageSet(
        cfg["DATA_DIR"],
        cfg["DATASET_STRUCTURE"],
        phase="testing",
        augmentations=build_sem_seg_train_aug(),
        give_field=True,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=cfg["NUM_WORKERS"],
    )

    for j, data_point in enumerate(data_loader):
        if j % 100 == 0:
            print("Training set {} of {}".format(j, len(data_set)))

        img = data_point["image"][0]
        instance = data_point["instance"][0]
        angle = data_point["angle"][0]
        dt = data_point["dt"][0]

        encoded_instance = torch.stack(
            (instance, torch.zeros_like(instance), torch.zeros_like(instance)), dim=-1
        )
        encoded_angle = np.array(convert_representation(angle)).astype(np.uint8)
        encoded_dt = np.array(convert_representation(dt, min=0, max=4000)).astype(
            np.uint8
        )

        img_im = Image.fromarray(np.array(img).astype(np.uint8))
        instance_im = Image.fromarray(np.array(encoded_instance).astype(np.uint8))
        angle_im = Image.fromarray(encoded_angle)
        dt_im = Image.fromarray(encoded_dt)

        img_im.save(data_point["image_file_name"][0])
        instance_im.save(data_point["instance_file_name"][0])
        angle_im.save(data_point["angle_file_name"][0])
        dt_im.save(data_point["dt_file_name"][0])

    return


if __name__ == "__main__":
    main()
