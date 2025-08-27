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

import yaml

import warnings

warnings.filterwarnings("ignore")


parser = ArgumentParser()

parser.add_argument(
    "--cfg",
    type=str,
    default="configs/New_PolypVideoDataset.yaml",
    help="Configuration file to use",
)
parser.add_argument(
    "--checkpoint",
    type=str,
# 一个数据集的
#     default="checkpoints/xxx/xx/New_PolypVideoDataset_23/New_PolypVideoDataset_23.ckpt",
    # default="checkpoints/xxx/xx/New_PolypVideoDataset_20/New_PolypVideoDataset_20.ckpt",另外一个数据集
    default="checkpoints/xxx/xx/New_PolypVideoDataset_15/New_PolypVideoDataset_15.ckpt",
    # 新视频的

    help="Path to model checkpoint (if None, loads the latest)",
)

train_opt = parser.parse_args()

with open(train_opt.cfg) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

# 在这里添加配置修改代码
if cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
    print("配置视频数据集以返回完整序列...")
    # 修改配置，使build_data创建完整序列的数据集
    cfg["VIDEO_TEST"] = {"clip_len": None}

# train_opt.isTrain = True
# train_opt.save_pth_dir = make_dir(cfg)

train_opt.isTrain = True

train_opt.save_pth_dir = make_dir(cfg)
train_set, val_set = build_data(cfg)

train_opt.class_list = val_set.CLASSES

if cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
    print("重新创建视频数据集以返回完整序列...")
    from dataloader.New_PolypVideoDataset import New_PolypVideoDataset

    dataset_root = cfg["DATASET"]["dataroot"]
    test_set = New_PolypVideoDataset(  # 使用test_set作为变量名
        cfg,
        root=dataset_root,
        split="test",  # 修改为test而不是val
        clip_len=None  # 返回完整序列
    )
    print(f"视频测试数据集包含 {len(test_set)} 个视频序列")
    train_opt.class_list = test_set.CLASSES  # 更新类别列表

    # 使用test_set替换val_set
    val_set = test_set  # 这样下面创建data_loader_val时就会使用测试集





data_loader_train = DataLoader(
    train_set,
    batch_size=cfg["TRAIN"]["batch_size"],
    num_workers=cfg["TRAIN"]["num_workers"],
    pin_memory=True,
    shuffle=True,
)

data_loader_val = DataLoader(
    val_set, batch_size=1, num_workers=cfg["TRAIN"]["num_workers"], pin_memory=True
)

train_opt.total_samples = len(data_loader_train)

model = build_model(train_opt, cfg)


trainer = pl.Trainer(
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    logger=False,
    enable_model_summary=False,
)

trainer.test(model=model, dataloaders=data_loader_val, ckpt_path=train_opt.checkpoint)




# import os
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import pytorch_lightning as pl
#
# from torch.utils.data import DataLoader
#
# from argparse import ArgumentParser
# import numpy as np
#
# from util.util import *
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.callbacks import LearningRateFinder
#
# import yaml
#
# import warnings
#
# warnings.filterwarnings("ignore")
#
#
# parser = ArgumentParser()
#
# parser.add_argument(
#     "--cfg",
#     type=str,
#     default="configs/New_PolypDataset.yaml",
#     help="Configuration file to use",
# )
# parser.add_argument(
#     "--checkpoint",
#     type=str,
#     default="checkpoints/xxx/xx/Polyp_Benchmark/Polyp_Benchmark.ckpt",
#     help="Path to model checkpoint (if None, loads the latest)",
# )
#
# train_opt = parser.parse_args()
#
# with open(train_opt.cfg) as f:
#     cfg = yaml.load(f, Loader=yaml.SafeLoader)
#
# train_opt.isTrain = True
#
# train_opt.save_pth_dir = make_dir(cfg)
# train_set, val_set = build_data(cfg)
#
# train_opt.class_list = val_set.CLASSES
#
# data_loader_train = DataLoader(
#     train_set,
#     batch_size=cfg["TRAIN"]["batch_size"],
#     num_workers=cfg["TRAIN"]["num_workers"],
#     pin_memory=True,
#     shuffle=True,
# )
#
# data_loader_val = DataLoader(
#     val_set, batch_size=1, num_workers=cfg["TRAIN"]["num_workers"], pin_memory=True
# )
#
# train_opt.total_samples = len(data_loader_train)
#
# model = build_model(train_opt, cfg)
#
# trainer = pl.Trainer(
#     devices=1,
#     accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     logger=False,
#     enable_model_summary=False,
# )
#
# trainer.test(model=model, dataloaders=data_loader_val, ckpt_path=train_opt.checkpoint)
