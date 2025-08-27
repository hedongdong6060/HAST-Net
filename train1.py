import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from argparse import ArgumentParser
import numpy as np

from util.util import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateFinder

from dataloader.New_PolypVideoDataset import (
    RandomClipSampler,
    SlidingWindowClipSampler,
)


import yaml

import warnings
# 在导入语句之前添加
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings("ignore")

torch.cuda.set_per_process_memory_fraction(1.0, 0)

parser = ArgumentParser()

parser.add_argument(
    "--cfg",
    type=str,
    default="./configs/New_PolypVideoDataset.yaml",
    help="Configuration file to use",
)

train_opt = parser.parse_args()

with open(train_opt.cfg) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

train_opt.isTrain = True

train_opt.save_pth_dir = make_dir(cfg)
train_set, val_set = build_data(cfg)
train_opt.class_list = val_set.CLASSES

if cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
    sampler = RandomClipSampler(dataset=train_set, clip_len=8, stride=8, drop_last=True)

    data_loader_train = DataLoader(
        dataset=train_set,
        sampler=sampler,
        batch_size=cfg["TRAIN"]["batch_size"],
        num_workers=0,
        pin_memory=True,
    )
    # 【在这里添加调试代码】
    print("\n==== 数据加载器统计 ====")
    print(f"批次数量: {len(data_loader_train)}")
    print(f"批次大小: {cfg['TRAIN']['batch_size']}")
    print(f"预计总样本数: {len(data_loader_train) * cfg['TRAIN']['batch_size']}")

    # 检查实际加载情况
    batch_count = 0
    video_samples = set()
    for sample in data_loader_train:
        batch_count += 1
        folders = sample["folder_name"]
        for folder in folders:
            video_samples.add(folder)

        if batch_count % 10 == 0:
            print(f"已处理 {batch_count} 批次，覆盖 {len(video_samples)} 个视频")

    print(f"总批次: {batch_count}，总样本: {batch_count * cfg['TRAIN']['batch_size']}")
    print(f"覆盖视频数量: {len(video_samples)}/{len(train_set.subfolders)}")

    sampler = SlidingWindowClipSampler(
        dataset=val_set, clip_len=8, stride=8, drop_last=False
    )

    data_loader_val = DataLoader(
        dataset=val_set,
        sampler=sampler,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )


else:
    data_loader_train = DataLoader(
        train_set,
        batch_size=cfg["TRAIN"]["batch_size"],
        num_workers=cfg["TRAIN"]["num_workers"],
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        val_set, batch_size=1, num_workers=cfg["TRAIN"]["num_workers"], pin_memory=True
    )

train_opt.total_samples = len(data_loader_train)

checkpoint_callback = ModelCheckpoint(
    monitor="index/average_IoU",
    dirpath=train_opt.save_pth_dir,
    filename=cfg["DATASET"]["name"],
    mode="max",
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

tensorboard_logger = TensorBoardLogger(save_dir=train_opt.save_pth_dir)
#


model = build_model(train_opt, cfg)

trainer = pl.Trainer(
    # strategy=DDPStrategy(),
    devices=cfg["TRAIN"]["node"],
    max_epochs=cfg["TRAIN"]["nepoch"],
    callbacks=[checkpoint_callback, lr_monitor],
    default_root_dir=train_opt.save_pth_dir,
    logger=tensorboard_logger,
    log_every_n_steps=50,
    check_val_every_n_epoch=cfg["TRAIN"]["eval_interval"],
    enable_model_summary=False,
)

if not cfg["continue_train"]:
    trainer.fit(
        model=model,
        train_dataloaders=data_loader_train,
        val_dataloaders=data_loader_val,
    )
else:
    pth_dir = get_ckpt_file(train_opt.save_pth_dir)
    trainer.fit(
        model=model,
        train_dataloaders=data_loader_train,
        val_dataloaders=data_loader_val,
        ckpt_path=pth_dir,
    )
