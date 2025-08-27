import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import cv2
import warnings
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import random


from dataloader import custom_transforms as tr


class New_PolypDataset(data.Dataset):
    """
    兼容两种采样方式(随机/滑窗)的 Dataset。
    - 当 clip_len is None 时，默认取完整序列；
    - 当 clip_len is not None, 需要 Sampler 提供 (video_idx, start_idx) 以截取子序列的一部分。
    """

    CLASSES = ["background", "Polyp", "instrument"]

    def __init__(self, cfg, root="", split="train", clip_len=None):
        """
        Args:
            cfg: 配置字典, 包含 TRAIN/TEST/数据增强 等信息
            root: 数据根目录, 里面包含 images/<split> 和 mask/<split>
            split: "train"/"val"/"test"
            clip_len: 如果为 None, 表示取整段; 否则取连续 clip_len 帧
        """
        self.root = root
        self.split = split
        self.cfg = cfg
        self.images = {}
        self.labels = {}

        self.image_base = os.path.join(root, "images", split)
        self.label_base = os.path.join(root, "mask", split)


        self.images[split] = []
        self.images[split] = self.recursive_glob(rootdir=self.image_base, suffix=".jpg")
        self.images[split].sort()

        self.labels[split] = []
        self.labels[split] = self.recursive_glob(rootdir=self.label_base, suffix=".png")
        self.labels[split].sort()

        if not self.images[split]:
            raise Exception(
                "No RGB images for split=[%s] found in %s" % (split, self.image_base)
            )

        if not self.labels[split]:
            raise Exception(
                "No labels for split=[%s] found in %s" % (split, self.label_base)
            )

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s label images" % (len(self.labels[split]), split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):
        img_path = self.images[self.split][index].rstrip()
        lbl_path = self.labels[self.split][index].rstrip()

        rgb_image = cv2.imread(img_path)
        label_image = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        oriHeight, oriWidth = label_image.shape

        sample = {"image": rgb_image, "label": label_image}

        self.path = img_path

        if self.split == "train":
            sample = self.transform_tr(sample)
        elif self.split == "val":
            sample = self.transform_val(sample)
        else:
            sample = self.transform_ts(sample)

        sample["img_path"] = img_path
        sample["oriHeight"] = oriHeight
        sample["oriWidth"] = oriWidth
        sample["oriSize"] = (oriHeight, oriWidth)

        sample["name"] = "/".join(img_path.rsplit("/", 2)[1:])

        return sample

    def recursive_glob(self, rootdir=".", suffix=""):
        return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.WaveletTransform(),
                tr.Resize(
                    image_size=(
                        self.cfg["TRAIN"]["size"][1],
                        self.cfg["TRAIN"]["size"][0],
                    ),
                    other_size=(128, 128),
                ),
                tr.Normalize_tensor(
                    Low_mean=self.cfg["DATASET"]["Low_mean"],
                    Low_std=self.cfg["DATASET"]["Low_std"],
                    High_mean=self.cfg["DATASET"]["High_mean"],
                    High_std=self.cfg["DATASET"]["High_std"],
                ),
            ]
        )
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.WaveletTransform(),
                tr.Resize(
                    image_size=(
                        self.cfg["TRAIN"]["size"][1],
                        self.cfg["TRAIN"]["size"][0],
                    ),
                    other_size=(128, 128),
                ),
                tr.Normalize_tensor(
                    Low_mean=self.cfg["DATASET"]["Low_mean"],
                    Low_std=self.cfg["DATASET"]["Low_std"],
                    High_mean=self.cfg["DATASET"]["High_mean"],
                    High_std=self.cfg["DATASET"]["High_std"],
                ),
            ]
        )
        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.WaveletTransform(),
                tr.Resize(
                    image_size=(
                        self.cfg["TRAIN"]["size"][1],
                        self.cfg["TRAIN"]["size"][0],
                    ),
                    other_size=(128, 128),
                ),
                tr.Normalize_tensor(
                    Low_mean=self.cfg["DATASET"]["Low_mean"],
                    Low_std=self.cfg["DATASET"]["Low_std"],
                    High_mean=self.cfg["DATASET"]["High_mean"],
                    High_std=self.cfg["DATASET"]["High_std"],
                ),
            ]
        )
        return composed_transforms(sample)
