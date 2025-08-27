import os
import glob
import cv2
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms

import math
import random
from torch.utils.data import Sampler

import sys


sys.path.append("/home/siyu/polyp_miccai")
# from dataloader import custom_transforms as tr

from dataloader import enhanced_transforms as tr
class New_PolypVideoDataset(data.Dataset):
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
        self.cfg = cfg
        self.split = split
        self.root = root
        self.clip_len = clip_len

        # 目录组织:  root/images/<split>/<subfolder>/*.jpg
        #           root/mask/<split>/<subfolder>/*.png
        self.image_root = os.path.join(root, "images", split)
        self.mask_root = os.path.join(root, "mask", split)

        # 列举所有子目录(子序列)
        self.subfolders = sorted(
            [
                d
                for d in os.listdir(self.image_root)
                if os.path.isdir(os.path.join(self.image_root, d))
            ],
            key=lambda x: int(x),  # 按子目录名称数字排序(假设子目录名是数字字符串)
        )
        if not self.subfolders:
            raise FileNotFoundError(f"No subfolders found in {self.image_root}")

        # 预先把每个子序列的所有帧路径收集起来, 以便在 __getitem__ 时方便索引
        # video_info[i] = {
        #   "folder_name": 该子序列目录名,
        #   "image_paths": [...],
        #   "mask_paths":  [...],
        #   "num_frames":  帧数
        # }
        self.video_info = []
        for folder_name in self.subfolders:
            image_folder = os.path.join(self.image_root, folder_name)
            mask_folder = os.path.join(self.mask_root, folder_name)
            image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
            mask_paths = sorted(glob.glob(os.path.join(mask_folder, "*.png")))
            if len(image_paths) != len(mask_paths):
                raise ValueError(
                    f"Images/Masks count mismatch in folder {folder_name}."
                )
            if not image_paths:
                raise FileNotFoundError(f"No images found in {image_folder}")

            self.video_info.append(
                {
                    "folder_name": folder_name,
                    "image_paths": image_paths,
                    "mask_paths": mask_paths,
                    "num_frames": len(image_paths),
                }
            )

    def __len__(self):
        """
        返回子序列的数量。
        Sampler 会对这些索引进行组合(如 (video_idx, start_idx))，从而决定实际返回多少个 clip。
        """
        return len(self.video_info)



    def __getitem__(self, index_or_tuple):
        """
        根据 Sampler 的输出, 我们可能收到:
          - 仅仅是一个 int (index) => 表示要整个子序列(clip_len=None) 或默认处理
          - 一个 (video_idx, start_idx) => 表示要在该子序列里从 start_idx 开始, 取 clip_len 帧

        注意 Sampler 会把 (video_idx, start_idx) 作为一个 tuple 传进来。
        """
        if isinstance(index_or_tuple, tuple):
            video_idx, start_idx = index_or_tuple
        else:
            # 如果不是 tuple, 则只传了一个 int => 默认从0开始
            video_idx = index_or_tuple
            start_idx = 0

        folder_name = self.video_info[video_idx]["folder_name"]
        image_paths = self.video_info[video_idx]["image_paths"]
        mask_paths = self.video_info[video_idx]["mask_paths"]
        num_frames = self.video_info[video_idx]["num_frames"]

        # 根据 clip_len 取相应的连续帧
        if self.clip_len is None:
            # clip_len=None => 取完整序列
            select_image_paths = image_paths
            select_mask_paths = mask_paths
        else:
            end_idx = start_idx + self.clip_len
            if end_idx > num_frames:
                end_idx = num_frames  # 或者做补帧
            select_image_paths = image_paths[start_idx:end_idx]
            select_mask_paths = mask_paths[start_idx:end_idx]

        # 读取并转换
        high_freq_list = []
        low_freq_list = []
        mask_list = []
        img_paths_list = []
        names_list = []

        for img_path, msk_path in zip(select_image_paths, select_mask_paths):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

            sample = {"image": img, "label": mask}

            if self.split == "train":
                sample = self.transform_tr(sample)
            else:
                # 这里也可区分 val/test, 如果有不同的预处理
                sample = self.transform_ts(sample)

            high_freq_list.append(sample["high_freq"])
            low_freq_list.append(sample["low_freq"])
            mask_list.append(sample["label"])
            img_paths_list.append(img_path)
            names_list.append(os.path.join(folder_name, os.path.basename(img_path)))

        # 堆叠成 (T, C, H, W) 或 (T, H, W)
        high_freq_tensor = torch.stack(high_freq_list, dim=0)
        low_freq_tensor = torch.stack(low_freq_list, dim=0)
        mask_tensor = torch.stack(mask_list, dim=0)

        out_sample = {
            "high_freq": high_freq_tensor,  # (T, C, H, W)
            "low_freq": low_freq_tensor,  # (T, C, H, W)
            "label": mask_tensor,  # (T, H, W) or (T,1,H,W) 看你的 transform
            "img_paths": img_paths_list,
            "names": names_list,
            "folder_name": folder_name,
            "start_idx": start_idx,  # 方便后面做后处理融合时知道这个clip对应的起点
        }
        return out_sample

    # # 在文件顶部导入
    # import sys
    # sys.path.append("/home/siyu/polyp_miccai")
    # from dataloader import enhanced_transforms as tr  # 修改这里
    #
    # # 然后修改transform_tr和transform_ts方法
    def transform_tr(self, sample):
        """
        训练阶段的 transforms
        """
        composed_transforms = transforms.Compose(
            [
                # 选择以下三种变换之一
                # tr.EnhancedWaveletTransform(),  # 简单的加权融合
                # tr.MultiScaleWaveletTransform(levels=2),  # 多尺度处理
                tr.FrequencyAttentionTransform(),  # 基于注意力的融合
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
        """
        测试/验证阶段的 transforms
        """
        composed_transforms = transforms.Compose(
            [
                # 选择与训练阶段相同的变换
                # tr.EnhancedWaveletTransform(),
                # tr.MultiScaleWaveletTransform(levels=2),
                tr.FrequencyAttentionTransform(),
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

    # ================== Transforms =====================
    # def transform_tr(self, sample):
    #     """
    #     训练阶段的 transforms
    #     """
    #     composed_transforms = transforms.Compose(
    #         [
    #             tr.WaveletTransform(),
    #             tr.Resize(
    #                 image_size=(
    #                     self.cfg["TRAIN"]["size"][1],
    #                     self.cfg["TRAIN"]["size"][0],
    #                 ),
    #                 other_size=(128, 128),
    #             ),
    #             tr.Normalize_tensor(
    #                 Low_mean=self.cfg["DATASET"]["Low_mean"],
    #                 Low_std=self.cfg["DATASET"]["Low_std"],
    #                 High_mean=self.cfg["DATASET"]["High_mean"],
    #                 High_std=self.cfg["DATASET"]["High_std"],
    #             ),
    #         ]
    #     )
    #     return composed_transforms(sample)
    #
    # def transform_ts(self, sample):
    #     """
    #     测试/验证阶段的 transforms
    #     """
    #     composed_transforms = transforms.Compose(
    #         [
    #             tr.WaveletTransform(),
    #             tr.Resize(
    #                 image_size=(
    #                     self.cfg["TRAIN"]["size"][1],
    #                     self.cfg["TRAIN"]["size"][0],
    #                 ),
    #                 other_size=(128, 128),
    #             ),
    #             tr.Normalize_tensor(
    #                 Low_mean=self.cfg["DATASET"]["Low_mean"],
    #                 Low_std=self.cfg["DATASET"]["Low_std"],
    #                 High_mean=self.cfg["DATASET"]["High_mean"],
    #                 High_std=self.cfg["DATASET"]["High_std"],
    #             ),
    #         ]
    #     )
    #     return composed_transforms(sample)


class RandomClipSampler(Sampler):
    """
    训练时使用的采样器：
    对每个视频子序列，采用滑窗方式提取所有连续片段，
    然后随机打乱整个采样列表。
    参数说明：
      - dataset: New_PolypVideoDataset 实例
      - clip_len: 每个片段的帧数
      - stride:   滑动步长
      - shuffle:  是否随机打乱所有采样项（默认 True）
      - drop_last: 如果子序列帧数小于 clip_len 时是否跳过（默认 True）
    """

    def __init__(self, dataset, clip_len=8, stride=8, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.video_info = dataset.video_info  # 每个子序列的元信息
        self.clip_len = clip_len
        self.stride = stride
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 预先生成所有 (video_idx, start_idx) 对
        self.clips = []
        for video_idx, info in enumerate(self.video_info):
            num_frames = info["num_frames"]
            if num_frames < self.clip_len:
                if self.drop_last:
                    continue
                else:
                    # 如果视频帧数不足，则取从0开始的完整序列
                    self.clips.append((video_idx, 0))
            else:
                max_start = num_frames - self.clip_len
                start_idx = 0
                while start_idx <= max_start:
                    self.clips.append((video_idx, start_idx))
                    start_idx += self.stride

    def __iter__(self):
        # 根据配置决定是否随机打乱所有采样项
        if self.shuffle:
            random.shuffle(self.clips)
        return iter(self.clips)

    def __len__(self):
        return len(self.clips)


class SlidingWindowClipSampler(Sampler):
    """
    测试/验证时，用滑窗方式遍历所有子序列。
    对于每个子序列, 在 [0 .. num_frames - clip_len] 内, 每 stride 个位置取一个window。
    """

    def __init__(self, dataset, clip_len=16, stride=8, shuffle=False, drop_last=False):
        """
        dataset: New_PolypVideoDataset
        clip_len: 窗口大小
        stride: 滑动步长
        shuffle: 是否打乱子序列顺序(不打乱帧内顺序)
        drop_last: 如果子序列比 clip_len 短, 是否直接丢弃
        """
        self.dataset = dataset
        self.video_info = dataset.video_info
        self.clip_len = clip_len
        self.stride = stride
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(self.video_info)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        for video_idx in self.indices:
            num_frames = self.video_info[video_idx]["num_frames"]
            if num_frames < self.clip_len:
                if self.drop_last:
                    # 跳过
                    continue
                else:
                    # 整段一次 (clip可能比实际帧数长)
                    yield (video_idx, 0)
                    continue

            max_start = num_frames - self.clip_len
            start_idx = 0
            while start_idx <= max_start:
                yield (video_idx, start_idx)
                start_idx += self.stride

    def __len__(self):
        total = 0
        for meta in self.video_info:
            T = meta["num_frames"]
            if T < self.clip_len:
                if not self.drop_last:
                    total += 1
            else:
                max_start = T - self.clip_len
                total_clips = math.floor(max_start / self.stride) + 1
                total += total_clips
        return total


if __name__ == "__main__":
    import yaml

    with open("configs/New_PolypVideoDataset.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    from torch.utils.data import DataLoader

    dataset_root = "dataset/New_PolypVideoDataset"
    split = "train"

    train_dataset = New_PolypVideoDataset(
        cfg, root=dataset_root, split="train", clip_len=8
    )
    # 2) Sampler
    sampler = RandomClipSampler(
        dataset=train_dataset, clip_len=8, stride=8, drop_last=True
    )

    # sampler = SlidingWindowClipSampler(
    #     dataset=train_dataset, clip_len=8, stride=8, drop_last=False
    # )
    # 3) DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=sampler,
        batch_size=3,
        num_workers=0,
        pin_memory=True,
    )
    print(len(train_loader))

    for sample in train_loader:
        # print(sample["high_freq"].shape)
        # print(sample["low_freq"].shape)
        # print(sample["masks"].shape)
        # print(sample["img_paths"])
        print(sample["names"])
        break




