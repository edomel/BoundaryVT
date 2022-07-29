import detectron2.data.transforms as T


def build_sem_seg_train_aug(data_cfg):
    augs = [
        T.ResizeShortestEdge(
            data_cfg["MIN_RESIZE"],
            data_cfg["MAX_SIZE"],
            data_cfg["MIN_SIZE_TRAIN_SAMPLING"],
        )
    ]
    if data_cfg["CROP"]["ENABLED"]:
        augs.append(T.RandomCrop(data_cfg["CROP"]["TYPE"], data_cfg["CROP"]["SIZE"]))
    augs.append(T.RandomFlip(prob=data_cfg["FLIP_PROB"]))
    return augs
