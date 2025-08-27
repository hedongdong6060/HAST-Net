import torch
import torch.nn as nn

import pytorch_lightning as pl
import torch.nn.functional as F
from util.scheduler import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.scheduler import CosineLRScheduler, PolyLRScheduler
import math
import torchmetrics
from util.metrics import *
from tabulate import tabulate
from pathlib import Path
import timm
import cv2

import torch.linalg as LA
import time


class seg_network(pl.LightningModule):

    def __init__(self, model, opt, cfg):
        super().__init__()
        self.model = model
        self.opt = opt
        self.cfg = cfg

        if len(cfg["MODEL"]["load_Pretraining"].strip()) != 0:
            # print("loading the model from %s" % cfg["MODEL"]["load_Pretraining"])
            checkpoint = torch.load(
                cfg["MODEL"]["load_Pretraining"], map_location="cpu"
            )
            pretrained_state_dict = checkpoint["state_dict"]

            if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":

                new_state_dict = {}
                for key, value in pretrained_state_dict.items():
                    new_key = "wavelet_net." + key
                    new_state_dict[new_key] = value

                self.model.load_state_dict(new_state_dict, strict=False)

                for name, param in self.model.named_parameters():
                    if name in new_state_dict:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                self.model.load_state_dict(pretrained_state_dict, strict=False)

                for name, param in self.model.named_parameters():
                    if name in pretrained_state_dict:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def training_step(self, batch, batch_idx):
        x = batch

        if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
            B, T, H, W = x["label"].shape

            out = self.model(x["low_freq"].unsqueeze(2), x["high_freq"].unsqueeze(2))

            out = out.view(B * T, out.size(2), out.size(3), out.size(4))

            label = x["label"].view(B * T, H, W)

            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

            loss = F.cross_entropy(
                out, label, ignore_index=self.cfg["DATASET"]["ignore_index"]
            )

            self.log(
                "Loss/validation_loss",
                loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=B,
            )

        else:

            _, H, W = x["label"].shape
            out = self.model(x["low_freq"].unsqueeze(1), x["high_freq"].unsqueeze(1))

            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

            loss = F.cross_entropy(
                out, x["label"], ignore_index=self.cfg["DATASET"]["ignore_index"]
            )

            self.log(
                "Loss/train_segloss",
                loss,
                on_step=True,
                sync_dist=True,
                batch_size=self.cfg["TRAIN"]["batch_size"],
                prog_bar=True,
            )

        return loss

    def on_validation_epoch_start(self):
        if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:

            self.CVC_300_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.CVC_ClinicDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.CVC_ColonDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.ETIS_LaribPolypDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.Kvasir_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

        else:
            self.metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

        self.eval_last_path = os.path.join(
            self.opt.save_pth_dir,
            "eval_last_{}.txt".format(self.cfg["DATASET"]["dataset"]),
        )
        with open(self.eval_last_path, "a") as f:
            f.write(
                "\n\n\n!!!!!! Starting validation for epoch {} !!!!!\n".format(
                    self.current_epoch
                )
            )

    def validation_step(self, batch, batch_idx):
        x = batch

        if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
            B, T, H, W = x["label"].shape

            out = self.model(x["low_freq"].unsqueeze(2), x["high_freq"].unsqueeze(2))

            out = out.view(B * T, out.size(2), out.size(3), out.size(4))

            label = x["label"].view(B * T, H, W)

            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

            loss = F.cross_entropy(
                out, label, ignore_index=self.cfg["DATASET"]["ignore_index"]
            )

            self.log(
                "Loss/validation_loss",
                loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=B,
            )

            out_softmax = out.softmax(dim=1)
            for i in range(B * T):
                pred_i = out_softmax[i].unsqueeze(0)
                label_i = label[i].unsqueeze(0)
                self.metrics.update(pred_i, label_i)

        else:
            _, H, W = x["label"].shape
            out = self.model(x["low_freq"].unsqueeze(1), x["high_freq"].unsqueeze(1))

            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

            loss = F.cross_entropy(
                out, x["label"], ignore_index=self.cfg["DATASET"]["ignore_index"]
            )

            self.log(
                "Loss/validation_loss",
                loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )

            out = out.softmax(dim=1)

            if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:

                if "CVC-300" in x["img_path"][0]:
                    self.CVC_300_metrics.update(out, x["label"])

                elif "CVC-ClinicDB" in x["img_path"][0]:
                    self.CVC_ClinicDB_metrics.update(out, x["label"])

                elif "CVC-ColonDB" in x["img_path"][0]:
                    self.CVC_ColonDB_metrics.update(out, x["label"])

                elif "ETIS-LaribPolypDB" in x["img_path"][0]:
                    self.ETIS_LaribPolypDB_metrics.update(out, x["label"])

                elif "Kvasir" in x["img_path"][0]:
                    self.Kvasir_metrics.update(out, x["label"])
            else:
                self.metrics.update(out, x["label"])
        return loss

    def on_validation_epoch_end(self):

        average_IoU = 0.0

        if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
            sem_index = self.CVC_300_metrics.compute()
            self.log(
                "index/CVC_300_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_300_Accuracy",
                sem_index["mACC"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_300_F1score", sem_index["mF1"], sync_dist=True, batch_size=1
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} CVC_300 images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            ############################################################

            sem_index = self.CVC_ClinicDB_metrics.compute()
            self.log(
                "index/CVC_ClinicDB_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_ClinicDB_Accuracy",
                sem_index["mACC"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_ClinicDB_F1score",
                sem_index["mF1"],
                sync_dist=True,
                batch_size=1,
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} CVC_ClinicDB images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            ############################################################

            sem_index = self.CVC_ColonDB_metrics.compute()
            self.log(
                "index/CVC_ColonDB_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_ColonDB_Accuracy",
                sem_index["mACC"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/CVC_ColonDB_F1score",
                sem_index["mF1"],
                sync_dist=True,
                batch_size=1,
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} CVC_ColonDB images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            ############################################################

            sem_index = self.ETIS_LaribPolypDB_metrics.compute()
            self.log(
                "index/ETIS_LaribPolypDB_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/ETIS_LaribPolypDB_Accuracy",
                sem_index["mACC"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/ETIS_LaribPolypDB_F1score",
                sem_index["mF1"],
                sync_dist=True,
                batch_size=1,
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} ETIS_LaribPolypDB images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            ############################################################

            sem_index = self.Kvasir_metrics.compute()
            self.log(
                "index/Kvasir_IOU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "index/Kvasir_Accuracy", sem_index["mACC"], sync_dist=True, batch_size=1
            )
            self.log(
                "index/Kvasir_F1score", sem_index["mF1"], sync_dist=True, batch_size=1
            )

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            average_IoU += sem_index["IOUs"][-1]

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n=====!!!!!===== Eval on {} Kvasir images =====!!!!!=====\n".format(
                        self.cfg["MODEL"]["model_names"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

            average_IoU = average_IoU / 5

            self.log(
                "index/average_IoU",
                average_IoU,
                sync_dist=True,
                batch_size=1,
                prog_bar=True,
            )

        else:
            sem_index = self.metrics.compute()
            self.log(
                "index/average_IoU",
                sem_index["mIOU"],
                sync_dist=True,
                batch_size=1,
                prog_bar=True,
            )
            self.log("index/Accuracy", sem_index["mACC"], sync_dist=True, batch_size=1)
            self.log("index/F1score", sem_index["mF1"], sync_dist=True, batch_size=1)

            table = {
                "Class": list(self.opt.class_list) + ["Mean"],
                "IoU": sem_index["IOUs"] + [sem_index["mIOU"]],
                "F1": sem_index["F1"] + [sem_index["mF1"]],
                "Acc": sem_index["ACC"] + [sem_index["mACC"]],
            }

            with open(self.eval_last_path, "a") as f:
                f.write(
                    "\n============== Eval on {} {} images =================\n".format(
                        self.cfg["MODEL"]["model_names"], self.cfg["DATASET"]["dataset"]
                    )
                )
                f.write("\n")
                print(tabulate(table, headers="keys"), file=f)

    def on_test_epoch_start(self):
        # 添加FPS计时器
        self.inference_times = []
        self.total_frames = 0
        self.batch_count = 0

        # 确保目录存在
        os.makedirs(self.opt.save_pth_dir, exist_ok=True)

        # 创建性能报告文件路径
        self.performance_path = os.path.join(
            self.opt.save_pth_dir,
            f"performance_{self.cfg['DATASET']['dataset']}.txt"
        )

        # 初始化指标对象
        if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
            self.CVC_300_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.CVC_ClinicDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.CVC_ColonDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.ETIS_LaribPolypDB_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

            self.Kvasir_metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()
        else:
            self.metrics = Metrics(
                self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]
            ).cuda()

        # 打印开始测试信息
        print(
            f"\n{'=' * 20} 开始测试 {self.cfg['MODEL']['model_names']} 在 {self.cfg['DATASET']['dataset']} 上 {'=' * 20}")

    def test_step(self, batch, batch_idx):
        x = batch
        self.batch_count += 1

        if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
            B, T, H, W = x["label"].shape
            self.total_frames += B * T

            # 确保之前的GPU操作都完成
            torch.cuda.synchronize()
            start_time = time.time()

            # 模型推理
            out = self.model(x["low_freq"].unsqueeze(2), x["high_freq"].unsqueeze(2))

            # 确保GPU操作完成后再停止计时
            torch.cuda.synchronize()
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # 后续处理
            out = out.view(B * T, out.size(2), out.size(3), out.size(4))
            label = x["label"].view(B * T, H, W)
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

            loss = F.cross_entropy(
                out, label, ignore_index=self.cfg["DATASET"]["ignore_index"]
            )

            out = out.softmax(dim=1)
            self.metrics.update(out, label)
        else:
            # 处理图像数据，标签形状为[B, H, W]
            _, H, W = x["label"].shape
            out = self.model(x["low_freq"].unsqueeze(1), x["high_freq"].unsqueeze(1))

            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

            loss = F.cross_entropy(
                out, x["label"], ignore_index=self.cfg["DATASET"]["ignore_index"]
            )

            out = out.softmax(dim=1)
            if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
                # 处理benchmark数据集...
                if "CVC-300" in x["img_path"][0]:
                    self.CVC_300_metrics.update(out, x["label"])
                # 其他数据集处理...
            else:
                self.metrics.update(out, x["label"])

        return loss





    # def test_step(self, batch, batch_idx):
    #     x = batch
    #
    #     _, H, W = x["label"].shape
    #     out = self.model(x["low_freq"].unsqueeze(1), x["high_freq"].unsqueeze(1))
    #
    #     out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    #
    #     loss = F.cross_entropy(
    #         out, x["label"], ignore_index=self.cfg["DATASET"]["ignore_index"]
    #     )
    #
    #     out = out.softmax(dim=1)
    #     if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
    #
    #         if "CVC-300" in x["img_path"][0]:
    #             self.CVC_300_metrics.update(out, x["label"])
    #
    #         elif "CVC-ClinicDB" in x["img_path"][0]:
    #             self.CVC_ClinicDB_metrics.update(out, x["label"])
    #
    #         elif "CVC-ColonDB" in x["img_path"][0]:
    #             self.CVC_ColonDB_metrics.update(out, x["label"])
    #
    #         elif "ETIS-LaribPolypDB" in x["img_path"][0]:
    #             self.ETIS_LaribPolypDB_metrics.update(out, x["label"])
    #
    #         elif "Kvasir" in x["img_path"][0]:
    #             self.Kvasir_metrics.update(out, x["label"])
    #     else:
    #         self.metrics.update(out, x["label"])
    #     return loss
    # 总帧数: {self.total_frames}

    def on_test_epoch_end(self):
        # 如果是视频数据集，计算并显示FPS性能指标
        if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
            total_time = sum(self.inference_times)
            avg_time = total_time / len(self.inference_times) if self.inference_times else 0
            fps = self.total_frames / total_time if total_time > 0 else 0

            performance_report = f"""
        {'=' * 60}
        性能指标报告 - {self.cfg['DATASET']['dataset']}
        {'=' * 60}
        测试总批次视频数: {self.batch_count}

        总推理时间: {total_time:.4f} 秒
        平均每批次推理时间: {avg_time * 1000:.2f} 毫秒
        平均每帧推理时间: {1000 * total_time / self.total_frames:.2f} 毫秒
        FPS (帧/秒): {fps:.2f}
        {'=' * 60}
        """

            print(performance_report)

            # 保存性能报告到文件
            with open(self.performance_path, 'w') as f:
                f.write(performance_report)

            # 计算分割指标
            sem_index = self.metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            mean_table = {
                "Metric": ["Mean IoU", "Mean F1", "Mean Acc"],
                "Value": [f"{sem_index['mIOU']:.4f}", f"{sem_index['mF1']:.4f}", f"{sem_index['mACC']:.4f}"]
            }

            # 打印分割指标
            table_rows = zip(*table.values())
            mean_rows = zip(*mean_table.values())

            print(f"\n{'=' * 20} 分割评估结果 {'=' * 20}")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))
            print(tabulate(mean_rows, headers=mean_table.keys(), tablefmt="grid"))

            # 将评估结果写入报告
            with open(self.performance_path, 'a') as f:
                f.write(f"\n{'=' * 20} 分割评估结果 {'=' * 20}\n")
                f.write(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))
                f.write("\n")
                f.write(tabulate(mean_rows, headers=mean_table.keys(), tablefmt="grid"))

        elif "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
        # if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
            sem_index = self.CVC_300_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== CVC_300 =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

            ############################################################

            sem_index = self.CVC_ClinicDB_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== CVC_ClinicDB =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

            ############################################################

            sem_index = self.CVC_ColonDB_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== CVC_ColonDB =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

            ############################################################

            sem_index = self.ETIS_LaribPolypDB_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== ETIS_LaribPolypDB =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

            ############################################################

            sem_index = self.Kvasir_metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print("\n============== Kvasir =================\n")
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

        else:
            sem_index = self.metrics.compute()

            table = {
                "Class": list(self.opt.class_list),
                "IoU": sem_index["IOUs"],
                "F1": sem_index["F1"],
                "Acc": sem_index["ACC"],
            }

            table_rows = zip(*table.values())
            print(tabulate(table_rows, headers=table.keys(), tablefmt="grid"))

    def predict_step(self, batch, batch_idx):
        """
        Processes a batch (clip) of video data for prediction.
        Expects batch items like 'low_freq', 'high_freq', 'label', 'img_paths'.
        Returns predictions and info for all frames in the clip.
        """
        # Ensure batch is a dictionary and contains required keys for video data
        if not isinstance(batch, dict) or not all(
                key in batch for key in ["low_freq", "high_freq", "label", "img_paths"]):
            raise ValueError(
                "Batch must be a dictionary containing 'low_freq', 'high_freq', 'label', and 'img_paths'.")

        x = batch
        # Get dimensions (Batch, Time, Channel, Height, Width)
        # Expect B=1 for prediction script based on pred.py DataLoader
        # print(f"Shape of x['low_freq'] in predict_step: {x['low_freq'].shape}")
        low_freq_processed = x["low_freq"].unsqueeze(2)
        high_freq_processed = x["high_freq"].unsqueeze(2)
        B, T, C, H_in, W_in = low_freq_processed.shape # Unpack from the *processed* tensor

        # B, T, C, H_in, W_in = x["low_freq"].shape
        # Label shape is (B, T, H, W) or (B, T, 1, H, W) depending on transforms
        label_shape = x["label"].shape
        H_lbl, W_lbl = label_shape[-2], label_shape[-1]  # Get spatial dimensions

        # Prepare model input consistent with train/val steps
        # Assuming model expects (B, T, 1, C, H, W) based on train_step unsqueeze(2)
        # Check channel dim C before unsqueezing if needed
        # If C is already 1, no need to unsqueeze dim 2, but model expects it.
        # Let's match the training/validation step explicitly:
        low_freq_input = x["low_freq"].unsqueeze(2)  # (B, T, 1, C, H_in, W_in)
        high_freq_input = x["high_freq"].unsqueeze(2)  # (B, T, 1, C, H_in, W_in)



        # Model inference
        out = self.model(low_freq_input, high_freq_input)
        # Assume model output shape is (B, T, NumClasses, H_out, W_out)

        # Reshape and interpolate like in validation_step
        num_classes = out.size(2)
        # Reshape to (B*T, NumClasses, H_out, W_out) for interpolation
        out = out.view(B * T, num_classes, out.size(3), out.size(4))
        # Interpolate to the label size (H_lbl, W_lbl)
        out = F.interpolate(out, size=(H_lbl, W_lbl), mode="bilinear", align_corners=False)

        # Get final predictions (argmax) -> Shape: (B*T, H_lbl, W_lbl)
        preds = out.softmax(dim=1).argmax(dim=1)  # Resulting preds are integer class labels

        # Flatten the list of image paths (list of lists -> single list)
        # Handles potential batch_size > 1, though pred.py uses B=1
        # The dataloader returns a list of paths directly for B=1, T=clip_len
        # If B > 1, dataloader might stack them, need careful checking.
        # Assuming B=1 from pred.py, x["img_paths"] is already a flat list of T paths.
        img_paths_flat = x["img_paths"]

        # Reshape frequency maps to (B*T, C, H_in, W_in) to match preds and paths
        low_freq_flat = x["low_freq"].view(B * T, C, H_in, W_in)
        high_freq_flat = x["high_freq"].view(B * T, C, H_in, W_in)

        # Return dictionary containing predictions and info for all frames in the batch/clip
        return {
            "preds": preds,  # Tensor shape (T, H_lbl, W_lbl) because B=1
            "img_paths": img_paths_flat,  # List of T strings
            "low_freqs": low_freq_flat,  # Tensor shape (T, C, H_in, W_in)
            "high_freqs": high_freq_flat  # Tensor shape (T, C, H_in, W_in)
        }

    # def predict_step(self, batch, batch_idx):
    #
    #     """
    #                Processes a batch (clip) for prediction.
    #                Input batch comes from DataLoader(New_PolypVideoDataset, sampler=...).
    #                """
    #     # Ensure batch is a dictionary and contains required keys for video data
    #     # The sampler ensures batch_size=1, so B=1 here.
    #     required_keys = ["low_freq", "high_freq", "label", "img_paths"]
    #     if not isinstance(batch, dict) or not all(key in batch for key in required_keys):
    #         missing_keys = [k for k in required_keys if k not in batch]
    #         raise ValueError(f"Batch must be a dictionary containing {required_keys}. Missing: {missing_keys}")
    #
    #     x = batch
    #     # Get dimensions (Batch, Time, Channel, Height, Width)
    #     # Expect B=1 for prediction script based on pred.py DataLoader setup
    #     B, T, C, H_in, W_in = x["low_freq"].shape
    #     # Label shape is (B, T, H_lbl, W_lbl)
    #     _, _, H_lbl, W_lbl = x["label"].shape
    #
    #     # Prepare model input consistent with train/val steps
    #     # unsqueeze(2) adds a singleton dimension maybe expected by the model?
    #     # Check your WaveletNetPlus input expectation. If it's (B*T, C, H, W), adjust here.
    #     # Assuming model expects (B, T, 1, C, H_in, W_in) based on train_step unsqueeze(2)
    #     # Or perhaps the model handles the time dimension internally? Let's follow train/val logic.
    #     low_freq_input = x["low_freq"].unsqueeze(2)  # Shape becomes (B, T, 1, C, H_in, W_in)
    #     high_freq_input = x["high_freq"].unsqueeze(2)  # Shape becomes (B, T, 1, C, H_in, W_in)
    #
    #     # Model inference
    #     # Assuming model output shape is (B, T, NumClasses, H_out, W_out)
    #     out = self.model(low_freq_input, high_freq_input)
    #
    #     # Reshape and interpolate like in validation_step
    #     num_classes = out.size(2)
    #     # Reshape to (B*T, NumClasses, H_out, W_out) for interpolation
    #     out = out.view(B * T, num_classes, out.size(3), out.size(4))
    #
    #     # Interpolate to the label size (H_lbl, W_lbl)
    #     out = F.interpolate(out, size=(H_lbl, W_lbl), mode="bilinear", align_corners=False)
    #
    #     # Get final predictions (argmax) -> Shape: (B*T, H_lbl, W_lbl)
    #     preds = out.softmax(dim=1).argmax(dim=1)  # No squeeze needed here
    #
    #     # Flatten the list of image paths if it's nested (it should be B=1, T=...)
    #     # Dataloader returns 'img_paths' as a list of T paths for the single clip in the batch
    #     img_paths_flat = x["img_paths"]  # Should already be a list of T paths because B=1
    #
    #     # Reshape frequency maps to (B*T, C, H_in, W_in) to match preds and paths
    #     low_freq_flat = x["low_freq"].view(B * T, C, H_in, W_in)
    #     high_freq_flat = x["high_freq"].view(B * T, C, H_in, W_in)
    #
    #     # Return dictionary containing predictions and info for all frames in the batch/clip
    #     return {
    #         "preds": preds,  # Tensor shape (T, H_lbl, W_lbl) since B=1
    #         "img_paths": img_paths_flat,  # List of T strings
    #         "low_freqs": low_freq_flat,  # Tensor shape (T, C, H_in, W_in)
    #         "high_freqs": high_freq_flat  # Tensor shape (T, C, H_in, W_in)
    #     }
        # Ensure batch is a dictionary and contains required keys
        # if not isinstance(batch, dict) or not all(key in batch for key in ["low_freq", "high_freq", "label"]):
        #     raise ValueError("Batch must be a dictionary containing 'low_freq', 'high_freq', and 'label'.")
        #
        # _, H, W = batch["label"].shape
        # out = self.model(batch["low_freq"].unsqueeze(1), batch["high_freq"].unsqueeze(1))
        #
        # out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        #
        # out = out.softmax(dim=1)
        # pred = out.argmax(dim=1).squeeze()
        #
        # return {
        #     "pred": pred,
        #     "img_path": batch["img_path"],
        #     "low_freq": batch["low_freq"],
        #     "high_freq": batch["high_freq"],
        # }

        # def predict_step(self, batch, batch_idx):
        #     """
        #     Processes a batch (clip) for prediction.
        #     Input batch comes from DataLoader(New_PolypVideoDataset, sampler=...).
        #     """
        #     # Ensure batch is a dictionary and contains required keys for video data
        #     # The sampler ensures batch_size=1, so B=1 here.
        #     required_keys = ["low_freq", "high_freq", "label", "img_paths"]
        #     if not isinstance(batch, dict) or not all(key in batch for key in required_keys):
        #         missing_keys = [k for k in required_keys if k not in batch]
        #         raise ValueError(f"Batch must be a dictionary containing {required_keys}. Missing: {missing_keys}")
        #
        #     x = batch
        #     # Get dimensions (Batch, Time, Channel, Height, Width)
        #     # Expect B=1 for prediction script based on pred.py DataLoader setup
        #     B, T, C, H_in, W_in = x["low_freq"].shape
        #     # Label shape is (B, T, H_lbl, W_lbl)
        #     _, _, H_lbl, W_lbl = x["label"].shape
        #
        #     # Prepare model input consistent with train/val steps
        #     # unsqueeze(2) adds a singleton dimension maybe expected by the model?
        #     # Check your WaveletNetPlus input expectation. If it's (B*T, C, H, W), adjust here.
        #     # Assuming model expects (B, T, 1, C, H_in, W_in) based on train_step unsqueeze(2)
        #     # Or perhaps the model handles the time dimension internally? Let's follow train/val logic.
        #     low_freq_input = x["low_freq"].unsqueeze(2)  # Shape becomes (B, T, 1, C, H_in, W_in)
        #     high_freq_input = x["high_freq"].unsqueeze(2)  # Shape becomes (B, T, 1, C, H_in, W_in)
        #
        #     # Model inference
        #     # Assuming model output shape is (B, T, NumClasses, H_out, W_out)
        #     out = self.model(low_freq_input, high_freq_input)
        #
        #     # Reshape and interpolate like in validation_step
        #     num_classes = out.size(2)
        #     # Reshape to (B*T, NumClasses, H_out, W_out) for interpolation
        #     out = out.view(B * T, num_classes, out.size(3), out.size(4))
        #
        #     # Interpolate to the label size (H_lbl, W_lbl)
        #     out = F.interpolate(out, size=(H_lbl, W_lbl), mode="bilinear", align_corners=False)
        #
        #     # Get final predictions (argmax) -> Shape: (B*T, H_lbl, W_lbl)
        #     preds = out.softmax(dim=1).argmax(dim=1)  # No squeeze needed here
        #
        #     # Flatten the list of image paths if it's nested (it should be B=1, T=...)
        #     # Dataloader returns 'img_paths' as a list of T paths for the single clip in the batch
        #     img_paths_flat = x["img_paths"]  # Should already be a list of T paths because B=1
        #
        #     # Reshape frequency maps to (B*T, C, H_in, W_in) to match preds and paths
        #     low_freq_flat = x["low_freq"].view(B * T, C, H_in, W_in)
        #     high_freq_flat = x["high_freq"].view(B * T, C, H_in, W_in)
        #
        #     # Return dictionary containing predictions and info for all frames in the batch/clip
        #     return {
        #         "preds": preds,  # Tensor shape (T, H_lbl, W_lbl) since B=1
        #         "img_paths": img_paths_flat,  # List of T strings
        #         "low_freqs": low_freq_flat,  # Tensor shape (T, C, H_in, W_in)
        #         "high_freqs": high_freq_flat  # Tensor shape (T, C, H_in, W_in)
        #     }

        # ... (其他 imports 和 seg_network 的 __init__, training_step, validation_step, test_step 等保持不变) ...

        # class seg_network(pl.LightningModule):
            # ... (__init__, training_step, validation_step, test_step, etc.) ...

            # def predict_step(self, batch, batch_idx):  # <--- 注释掉这个单图版本
            #     # Ensure batch is a dictionary and contains required keys
            #     if not isinstance(batch, dict) or not all(key in batch for key in ["low_freq", "high_freq", "label"]):
            #         raise ValueError("Batch must be a dictionary containing 'low_freq', 'high_freq', and 'label'.")
            #
            #     _, H, W = batch["label"].shape
            #     out = self.model(batch["low_freq"].unsqueeze(1), batch["high_freq"].unsqueeze(1))
            #
            #     out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            #
            #     out = out.softmax(dim=1)
            #     pred = out.argmax(dim=1).squeeze()
            #
            #     return {
            #         "pred": pred,
            #         "img_path": batch["img_path"],
            #         "low_freq": batch["low_freq"],
            #         "high_freq": batch["high_freq"],
            #     }

            # --- 启用这个视频版本 ---


            # --- 视频版本 predict_step 结束 ---

            # def configure_optimizers(self):
        # ... (configure_optimizers 保持不变) ...
        # ... (文件末尾) ...

        # # ... inside class seg_network ...
        #
        # def predict_step(self, batch, batch_idx):
        #     # Ensure batch is a dictionary and contains required keys for video data
        #     if not isinstance(batch, dict) or not all(
        #             key in batch for key in ["low_freq", "high_freq", "label", "img_paths"]):
        #         raise ValueError(
        #             "Batch must be a dictionary containing 'low_freq', 'high_freq', 'label', and 'img_paths'.")
        #
        #     x = batch
        #     # Get dimensions (Batch, Time, Channel, Height, Width)
        #     # Expect B=1 for prediction script based on pred.py DataLoader
        #     B, T, C, H_in, W_in = x["low_freq"].shape
        #     _, _, H_lbl, W_lbl = x["label"].shape  # Label shape is (B, T, H, W)
        #
        #     # Prepare model input consistent with train/val steps
        #     # Assuming model expects (B, T, 1, C, H, W) based on train_step unsqueeze(2)
        #     low_freq_input = x["low_freq"].unsqueeze(2)
        #     high_freq_input = x["high_freq"].unsqueeze(2)
        #
        #     # Model inference
        #     out = self.model(low_freq_input, high_freq_input)
        #     # Assume model output shape is (B, T, NumClasses, H_out, W_out)
        #
        #     # Reshape and interpolate like in validation_step
        #     num_classes = out.size(2)
        #     # Reshape to (B*T, NumClasses, H_out, W_out) for interpolation
        #     out = out.view(B * T, num_classes, out.size(3), out.size(4))
        #     # Interpolate to the label size (H_lbl, W_lbl)
        #     out = F.interpolate(out, size=(H_lbl, W_lbl), mode="bilinear", align_corners=False)
        #
        #     # Get final predictions (argmax) -> Shape: (B*T, H_lbl, W_lbl)
        #     preds = out.softmax(dim=1).argmax(dim=1)
        #
        #     # Flatten the list of image paths (list of lists -> single list)
        #     # Handles batch_size > 1, though pred.py uses B=1
        #     img_paths_flat = [p for sublist in x["img_paths"] for p in sublist] if isinstance(x["img_paths"][0],
        #                                                                                       list) else x["img_paths"]
        #
        #     # Reshape frequency maps to (B*T, C, H_in, W_in) to match preds and paths
        #     low_freq_flat = x["low_freq"].view(B * T, C, H_in, W_in)
        #     high_freq_flat = x["high_freq"].view(B * T, C, H_in, W_in)
        #
        #     # Return dictionary containing predictions and info for all frames in the batch/clip
        #     return {
        #         "preds": preds,  # Tensor shape (B*T, H_lbl, W_lbl)
        #         "img_paths": img_paths_flat,  # List of B*T strings
        #         "low_freqs": low_freq_flat,  # Tensor shape (B*T, C, H_in, W_in)
        #         "high_freqs": high_freq_flat  # Tensor shape (B*T, C, H_in, W_in)
        #     }

        # def predict_step(self, batch, batch_idx):  <- Remove this old version
        #     x = batch
        #     _, H, W = x["label"].shape
        #     out = self.model(x["low_freq"].unsqueeze(1), x["high_freq"].unsqueeze(1))
        #
        #     out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        #
        #     out = out.softmax(dim=1)
        #     pred = out.argmax(dim=1).squeeze()
        #
        #     return {
        #         "pred": pred,
        #         "img_path": x["img_path"],
        #         "low_freq": x["low_freq"],
        #         "high_freq": x["high_freq"],
        #     }

        # def configure_optimizers(self):

    # ... existing code ...

    # def predict_step(self, batch, batch_idx):
    #     x = batch
    #     _, H, W = x["label"].shape
    #     out = self.model(x["low_freq"].unsqueeze(1), x["high_freq"].unsqueeze(1))
    #
    #     out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    #
    #     out = out.softmax(dim=1)
    #     pred = out.argmax(dim=1).squeeze()
    #
    #     return {
    #         "pred": pred,
    #         "img_path": x["img_path"],
    #         "low_freq": x["low_freq"],
    #         "high_freq": x["high_freq"],
    #     }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg["OPTIMIZER"]["lr"],
            betas=(0.9, 0.999),
            weight_decay=self.cfg["OPTIMIZER"]["weight_decay"],
        )

        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            self.cfg["TRAIN"]["nepoch"],
            self.cfg["SCHEDULER"]["warmup_epoch"],
            math.ceil(self.opt.total_samples / int(self.cfg["TRAIN"]["node"])),
            self.cfg["SCHEDULER"]["lr_warmup"],
            self.cfg["SCHEDULER"]["warmup_ratio"],
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler_config)































# # FILE: build_model.py
#
# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import torch.nn.functional as F
# from util.scheduler import *
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from timm.scheduler import CosineLRScheduler, PolyLRScheduler
# import math
# import torchmetrics
# from util.metrics import *
# from tabulate import tabulate
# from pathlib import Path
# import timm
# import cv2
# import torch.linalg as LA
# import time
# import os # Ensure os is imported
#
# # --- (Keep other imports and the seg_network class definition) ---
#
# class seg_network(pl.LightningModule):
#
#     def __init__(self, model, opt, cfg):
#         super().__init__()
#         self.model = model
#         self.opt = opt
#         self.cfg = cfg
#
#         # --- (Keep the rest of __init__ as is) ---
#         if len(cfg["MODEL"]["load_Pretraining"].strip()) != 0:
#             print("loading the model from %s" % cfg["MODEL"]["load_Pretraining"])
#             checkpoint = torch.load(
#                 cfg["MODEL"]["load_Pretraining"], map_location="cpu"
#             )
#             pretrained_state_dict = checkpoint["state_dict"]
#
#             if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
#
#                 new_state_dict = {}
#                 for key, value in pretrained_state_dict.items():
#                     # Check if the key already starts with 'wavelet_net.'
#                     # This might happen if loading a checkpoint already saved from WaveletNetPlus
#                     if not key.startswith('wavelet_net.'):
#                          new_key = "wavelet_net." + key
#                     else:
#                          new_key = key # Keep original key if already prefixed
#                     new_state_dict[new_key] = value
#
#                 # Load into the self.model which is WaveletNetPlus
#                 # Allow missing keys if the loaded checkpoint is only for wavelet_net part
#                 missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
#                 print("Loading Pretrained Weights:")
#                 if missing_keys:
#                     print("Missing keys:", missing_keys) # Likely LSTM/Attention parts
#                 if unexpected_keys:
#                     print("Unexpected keys:", unexpected_keys) # Should be empty if prefixing is correct
#
#                 # Decide which parameters to freeze based on successful loading
#                 # Freeze only the base wavelet_net parameters if loaded successfully
#                 print("Freezing parameters based on loaded state dict...")
#                 for name, param in self.model.named_parameters():
#                     if name in new_state_dict: # If the parameter was in the loaded dict
#                         param.requires_grad = False
#                         # print(f"Froze: {name}")
#                     else:
#                         param.requires_grad = True
#                         # print(f"Trainable: {name}")
#             else: # Non-video case (keep original logic)
#                 self.model.load_state_dict(pretrained_state_dict, strict=False)
#                 print("Freezing parameters based on loaded state dict...")
#                 for name, param in self.model.named_parameters():
#                     if name in pretrained_state_dict:
#                         param.requires_grad = False
#                     else:
#                         param.requires_grad = True
#
#
#     def training_step(self, batch, batch_idx):
#         x = batch
#
#         if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
#             # Label shape is (B, T, H, W)
#             B, T, H, W = x["label"].shape
#             # Input freq maps shape (B, T, C, H_in, W_in)
#
#             # === MODIFICATION ===
#             # Remove .unsqueeze(2) - WaveletNetPlus expects (B, T, C, H, W)
#             out = self.model(x["low_freq"], x["high_freq"])
#             # Model output shape is (B, T, NumClasses, H_out, W_out)
#
#             num_classes = out.shape[2] # Get NumClasses from output
#             # Reshape for frame-wise processing
#             out = out.view(B * T, num_classes, out.size(3), out.size(4))
#             # Reshape label
#             label = x["label"].view(B * T, H, W)
#
#             # Interpolate to label size
#             out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
#
#             loss = F.cross_entropy(
#                 out, label.long(), ignore_index=self.cfg["DATASET"]["ignore_index"] # Ensure label is long
#             )
#
#             self.log(
#                 "Loss/train_loss", # Log train loss specifically
#                 loss,
#                 on_step=True,
#                 on_epoch=True,
#                 sync_dist=True,
#                 batch_size=B, # B is the batch size (num clips, usually 1)
#                 prog_bar=True
#             )
#
#         else: # Non-video case
#             # Label shape is (B, H, W)
#             _, H, W = x["label"].shape
#             # Input freq maps shape (B, C, H_in, W_in)
#
#             # === MODIFICATION (Potentially, depends on non-video model) ===
#             # If the non-video model also expects (B, C, H, W), remove unsqueeze(1)
#             # Assuming non-video model expects single image input (B, C, H, W)
#             # The dataloader for non-video likely returns (B, C, H, W), so no unsqueeze needed
#             out = self.model(x["low_freq"], x["high_freq"]) # Adjust if non-video model differs
#             # Model output shape (B, NumClasses, H_out, W_out)
#
#             out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
#
#             loss = F.cross_entropy(
#                 out, x["label"].long(), ignore_index=self.cfg["DATASET"]["ignore_index"] # Ensure label is long
#             )
#
#             self.log(
#                 "Loss/train_segloss",
#                 loss,
#                 on_step=True,
#                 sync_dist=True,
#                 batch_size=self.cfg["TRAIN"]["batch_size"],
#                 prog_bar=True,
#             )
#
#         return loss
#
#     def on_validation_epoch_start(self):
#         # --- (Keep this method as is, Metrics setup looks fine) ---
#         if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
#             self.CVC_300_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#             self.CVC_ClinicDB_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#             self.CVC_ColonDB_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#             self.ETIS_LaribPolypDB_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#             self.Kvasir_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#         else:
#             self.metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#
#         # Ensure save directory exists for eval logs
#         os.makedirs(self.opt.save_pth_dir, exist_ok=True)
#         self.eval_last_path = os.path.join(
#             self.opt.save_pth_dir,
#             "eval_last_{}.txt".format(self.cfg["DATASET"]["dataset"]),
#         )
#         # Avoid appending multiple headers if resuming
#         if self.current_epoch == 0:
#              with open(self.eval_last_path, "w") as f: # Use 'w' for the first epoch
#                  f.write(f"Validation Log for {self.cfg['DATASET']['dataset']}\n")
#         with open(self.eval_last_path, "a") as f:
#             f.write(
#                 f"\n\n\n!!!!!! Starting validation for epoch {self.current_epoch} !!!!!\n"
#             )
#
#
#     def validation_step(self, batch, batch_idx):
#         x = batch
#
#         if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
#             B, T, H, W = x["label"].shape
#
#             # === MODIFICATION ===
#             # Remove .unsqueeze(2)
#             out = self.model(x["low_freq"], x["high_freq"])
#             # Output (B, T, NumClasses, H_out, W_out)
#
#             num_classes = out.shape[2]
#             out = out.view(B * T, num_classes, out.size(3), out.size(4))
#             label = x["label"].view(B * T, H, W)
#
#             out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
#
#             loss = F.cross_entropy(
#                 out, label.long(), ignore_index=self.cfg["DATASET"]["ignore_index"] # Ensure label is long
#             )
#
#             self.log(
#                 "Loss/validation_loss",
#                 loss,
#                 on_step=False, # Log per epoch on validation
#                 on_epoch=True,
#                 sync_dist=True,
#                 batch_size=B,
#             )
#
#             # Update metrics frame by frame
#             out_softmax = out.softmax(dim=1)
#             # No loop needed, metrics class likely handles batch dimension (B*T)
#             self.metrics.update(out_softmax, label)
#
#
#         else: # Non-video case
#             _, H, W = x["label"].shape
#
#             # === MODIFICATION (Potentially) ===
#             # Remove unsqueeze(1) if model expects (B, C, H, W)
#             out = self.model(x["low_freq"], x["high_freq"]) # Adjust if needed
#
#             out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
#
#             loss = F.cross_entropy(
#                 out, x["label"].long(), ignore_index=self.cfg["DATASET"]["ignore_index"] # Ensure label is long
#             )
#
#             self.log(
#                 "Loss/validation_loss",
#                 loss,
#                 on_step=False,
#                 on_epoch=True,
#                 sync_dist=True,
#                 batch_size=1, # Assuming batch_size 1 for non-video val? Check config
#             )
#
#             out_softmax = out.softmax(dim=1)
#
#             # Update metrics based on dataset type (Benchmark or other)
#             if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
#                 # --- (Keep Benchmark logic as is) ---
#                 if "CVC-300" in x["img_path"][0]:
#                     self.CVC_300_metrics.update(out_softmax, x["label"])
#                 elif "CVC-ClinicDB" in x["img_path"][0]:
#                     self.CVC_ClinicDB_metrics.update(out_softmax, x["label"])
#                 elif "CVC-ColonDB" in x["img_path"][0]:
#                     self.CVC_ColonDB_metrics.update(out_softmax, x["label"])
#                 elif "ETIS-LaribPolypDB" in x["img_path"][0]:
#                     self.ETIS_LaribPolypDB_metrics.update(out_softmax, x["label"])
#                 elif "Kvasir" in x["img_path"][0]:
#                     self.Kvasir_metrics.update(out_softmax, x["label"])
#             else:
#                  self.metrics.update(out_softmax, x["label"])
#         # Return loss is optional in validation_step if only logging matters
#         # return loss
#
#     def on_validation_epoch_end(self):
#         # --- (Keep this method as is, Metric computation and logging look fine) ---
#         average_IoU = 0.0
#         if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
#             # --- CVC-300 ---
#             sem_index = self.CVC_300_metrics.compute()
#             self.log("index/CVC_300_IOU", sem_index["mIOU"], sync_dist=True)
#             self.log("index/CVC_300_Accuracy", sem_index["mACC"], sync_dist=True)
#             self.log("index/CVC_300_F1score", sem_index["mF1"], sync_dist=True)
#             table = {"Class": list(self.opt.class_list) + ["Mean"], "IoU": sem_index["IOUs"] + [sem_index["mIOU"]], "F1": sem_index["F1"] + [sem_index["mF1"]], "Acc": sem_index["ACC"] + [sem_index["mACC"]]}
#             average_IoU += sem_index["IOUs"][-1] # Assuming last IoU is the target class
#             with open(self.eval_last_path, "a") as f:
#                  f.write(f"\n=====!!!!!===== Eval on {self.cfg['MODEL']['model_names']} CVC_300 images =====!!!!!=====\n")
#                  f.write("\n")
#                  print(tabulate(table, headers="keys"), file=f)
#             self.CVC_300_metrics.reset()
#
#             # --- CVC-ClinicDB ---
#             sem_index = self.CVC_ClinicDB_metrics.compute()
#             self.log("index/CVC_ClinicDB_IOU", sem_index["mIOU"], sync_dist=True)
#             self.log("index/CVC_ClinicDB_Accuracy", sem_index["mACC"], sync_dist=True)
#             self.log("index/CVC_ClinicDB_F1score", sem_index["mF1"], sync_dist=True)
#             table = {"Class": list(self.opt.class_list) + ["Mean"], "IoU": sem_index["IOUs"] + [sem_index["mIOU"]], "F1": sem_index["F1"] + [sem_index["mF1"]], "Acc": sem_index["ACC"] + [sem_index["mACC"]]}
#             average_IoU += sem_index["IOUs"][-1]
#             with open(self.eval_last_path, "a") as f:
#                 f.write(f"\n=====!!!!!===== Eval on {self.cfg['MODEL']['model_names']} CVC_ClinicDB images =====!!!!!=====\n")
#                 f.write("\n")
#                 print(tabulate(table, headers="keys"), file=f)
#             self.CVC_ClinicDB_metrics.reset()
#
#             # --- CVC-ColonDB ---
#             sem_index = self.CVC_ColonDB_metrics.compute()
#             self.log("index/CVC_ColonDB_IOU", sem_index["mIOU"], sync_dist=True)
#             self.log("index/CVC_ColonDB_Accuracy", sem_index["mACC"], sync_dist=True)
#             self.log("index/CVC_ColonDB_F1score", sem_index["mF1"], sync_dist=True)
#             table = {"Class": list(self.opt.class_list) + ["Mean"], "IoU": sem_index["IOUs"] + [sem_index["mIOU"]], "F1": sem_index["F1"] + [sem_index["mF1"]], "Acc": sem_index["ACC"] + [sem_index["mACC"]]}
#             average_IoU += sem_index["IOUs"][-1]
#             with open(self.eval_last_path, "a") as f:
#                 f.write(f"\n=====!!!!!===== Eval on {self.cfg['MODEL']['model_names']} CVC_ColonDB images =====!!!!!=====\n")
#                 f.write("\n")
#                 print(tabulate(table, headers="keys"), file=f)
#             self.CVC_ColonDB_metrics.reset()
#
#             # --- ETIS-LaribPolypDB ---
#             sem_index = self.ETIS_LaribPolypDB_metrics.compute()
#             self.log("index/ETIS_LaribPolypDB_IOU", sem_index["mIOU"], sync_dist=True)
#             self.log("index/ETIS_LaribPolypDB_Accuracy", sem_index["mACC"], sync_dist=True)
#             self.log("index/ETIS_LaribPolypDB_F1score", sem_index["mF1"], sync_dist=True)
#             table = {"Class": list(self.opt.class_list) + ["Mean"], "IoU": sem_index["IOUs"] + [sem_index["mIOU"]], "F1": sem_index["F1"] + [sem_index["mF1"]], "Acc": sem_index["ACC"] + [sem_index["mACC"]]}
#             average_IoU += sem_index["IOUs"][-1]
#             with open(self.eval_last_path, "a") as f:
#                 f.write(f"\n=====!!!!!===== Eval on {self.cfg['MODEL']['model_names']} ETIS_LaribPolypDB images =====!!!!!=====\n")
#                 f.write("\n")
#                 print(tabulate(table, headers="keys"), file=f)
#             self.ETIS_LaribPolypDB_metrics.reset()
#
#             # --- Kvasir ---
#             sem_index = self.Kvasir_metrics.compute()
#             self.log("index/Kvasir_IOU", sem_index["mIOU"], sync_dist=True)
#             self.log("index/Kvasir_Accuracy", sem_index["mACC"], sync_dist=True)
#             self.log("index/Kvasir_F1score", sem_index["mF1"], sync_dist=True)
#             table = {"Class": list(self.opt.class_list) + ["Mean"], "IoU": sem_index["IOUs"] + [sem_index["mIOU"]], "F1": sem_index["F1"] + [sem_index["mF1"]], "Acc": sem_index["ACC"] + [sem_index["mACC"]]}
#             average_IoU += sem_index["IOUs"][-1]
#             with open(self.eval_last_path, "a") as f:
#                 f.write(f"\n=====!!!!!===== Eval on {self.cfg['MODEL']['model_names']} Kvasir images =====!!!!!=====\n")
#                 f.write("\n")
#                 print(tabulate(table, headers="keys"), file=f)
#             self.Kvasir_metrics.reset()
#
#             average_IoU = average_IoU / 5
#             self.log("index/average_IoU", average_IoU, sync_dist=True, prog_bar=True)
#
#         else: # Non-benchmark case
#             sem_index = self.metrics.compute()
#             self.log("index/average_IoU", sem_index["mIOU"], sync_dist=True, prog_bar=True)
#             self.log("index/Accuracy", sem_index["mACC"], sync_dist=True)
#             self.log("index/F1score", sem_index["mF1"], sync_dist=True)
#
#             table = {"Class": list(self.opt.class_list) + ["Mean"], "IoU": sem_index["IOUs"] + [sem_index["mIOU"]], "F1": sem_index["F1"] + [sem_index["mF1"]], "Acc": sem_index["ACC"] + [sem_index["mACC"]]}
#
#             with open(self.eval_last_path, "a") as f:
#                 f.write(f"\n============== Eval on {self.cfg['MODEL']['model_names']} {self.cfg['DATASET']['dataset']} images =================\n")
#                 f.write("\n")
#                 print(tabulate(table, headers="keys"), file=f)
#             self.metrics.reset()
#
#
#     def on_test_epoch_start(self):
#         # --- (Keep this method as is, FPS and Metrics setup look fine) ---
#         self.inference_times = []
#         self.total_frames = 0
#         self.batch_count = 0
#         os.makedirs(self.opt.save_pth_dir, exist_ok=True)
#         self.performance_path = os.path.join(self.opt.save_pth_dir, f"performance_{self.cfg['DATASET']['dataset']}.txt")
#
#         if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
#             self.CVC_300_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#             self.CVC_ClinicDB_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#             self.CVC_ColonDB_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#             self.ETIS_LaribPolypDB_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#             self.Kvasir_metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#         else:
#             self.metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).to(self.device)
#         print(f"\n{'=' * 20} 开始测试 {self.cfg['MODEL']['model_names']} 在 {self.cfg['DATASET']['dataset']} 上 {'=' * 20}")
#
#     def test_step(self, batch, batch_idx):
#         x = batch
#         self.batch_count += 1
#
#         if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
#             B, T, H, W = x["label"].shape
#             self.total_frames += B * T
#
#             torch.cuda.synchronize()
#             start_time = time.time()
#
#             # === MODIFICATION ===
#             # Remove .unsqueeze(2)
#             out = self.model(x["low_freq"], x["high_freq"])
#             # Output (B, T, NumClasses, H_out, W_out)
#
#             torch.cuda.synchronize()
#             inference_time = time.time() - start_time
#             self.inference_times.append(inference_time)
#
#             num_classes = out.shape[2]
#             out = out.view(B * T, num_classes, out.size(3), out.size(4))
#             label = x["label"].view(B * T, H, W)
#             out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
#
#             # Loss calculation is optional in test_step unless needed for monitoring
#             # loss = F.cross_entropy(out, label.long(), ignore_index=self.cfg["DATASET"]["ignore_index"])
#
#             out_softmax = out.softmax(dim=1)
#             self.metrics.update(out_softmax, label)
#
#         else: # Non-video case
#             # === (Keep existing non-video logic, maybe remove unsqueeze(1) as noted before) ===
#              _, H, W = x["label"].shape
#              # === MODIFICATION (Potentially) ===
#              out = self.model(x["low_freq"], x["high_freq"]) # Adjust if needed
#              out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
#              # loss = F.cross_entropy(out, x["label"].long(), ignore_index=self.cfg["DATASET"]["ignore_index"])
#              out_softmax = out.softmax(dim=1)
#              if "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
#                  if "CVC-300" in x["img_path"][0]: self.CVC_300_metrics.update(out_softmax, x["label"])
#                  elif "CVC-ClinicDB" in x["img_path"][0]: self.CVC_ClinicDB_metrics.update(out_softmax, x["label"])
#                  elif "CVC-ColonDB" in x["img_path"][0]: self.CVC_ColonDB_metrics.update(out_softmax, x["label"])
#                  elif "ETIS-LaribPolypDB" in x["img_path"][0]: self.ETIS_LaribPolypDB_metrics.update(out_softmax, x["label"])
#                  elif "Kvasir" in x["img_path"][0]: self.Kvasir_metrics.update(out_softmax, x["label"])
#              else:
#                  self.metrics.update(out_softmax, x["label"])
#         # Return value from test_step is generally not used by Trainer
#         # return loss
#
#
#     def on_test_epoch_end(self):
#         # --- (Keep this method as is, FPS and Metric computation/saving look fine) ---
#          if self.cfg["DATASET"]["dataset"] == "New_PolypVideoDataset":
#             total_time = sum(self.inference_times)
#             avg_batch_time = total_time / len(self.inference_times) if self.inference_times else 0
#             fps = self.total_frames / total_time if total_time > 0 else 0
#             avg_frame_time_ms = (1000 * total_time / self.total_frames) if self.total_frames > 0 else 0
#
#             performance_report = f"""
#         {'=' * 60}
#         性能指标报告 - {self.cfg['DATASET']['dataset']}
#         {'=' * 60}
#         总批次数 (Clips): {self.batch_count}
#         总帧数: {self.total_frames}
#         总推理时间: {total_time:.4f} 秒
#         平均每批次(Clip)推理时间: {avg_batch_time * 1000:.2f} 毫秒
#         平均每帧推理时间: {avg_frame_time_ms:.2f} 毫秒
#         FPS (帧/秒): {fps:.2f}
#         {'=' * 60}
#         """
#             print(performance_report)
#             with open(self.performance_path, 'w') as f:
#                 f.write(performance_report)
#
#             sem_index = self.metrics.compute()
#             # Use correct class list from self.opt
#             table = {"Class": list(self.opt.class_list), "IoU": sem_index["IOUs"], "F1": sem_index["F1"], "Acc": sem_index["ACC"]}
#             mean_table = {"Metric": ["Mean IoU", "Mean F1", "Mean Acc"], "Value": [f"{sem_index['mIOU']:.4f}", f"{sem_index['mF1']:.4f}", f"{sem_index['mACC']:.4f}"]}
#
#             # Convert list of lists/tensors to printable format
#             iou_strs = [f"{iou:.4f}" for iou in table['IoU']]
#             f1_strs = [f"{f1:.4f}" for f1 in table['F1']]
#             acc_strs = [f"{acc:.4f}" for acc in table['Acc']]
#             table_rows = list(zip(table['Class'], iou_strs, f1_strs, acc_strs))
#             mean_rows = list(zip(mean_table['Metric'], mean_table['Value']))
#
#             print(f"\n{'=' * 20} 分割评估结果 {'=' * 20}")
#             print(tabulate(table_rows, headers=["Class", "IoU", "F1", "Acc"], tablefmt="grid"))
#             print(tabulate(mean_rows, headers=mean_table.keys(), tablefmt="grid"))
#
#             with open(self.performance_path, 'a') as f:
#                 f.write(f"\n{'=' * 20} 分割评估结果 {'=' * 20}\n")
#                 f.write(tabulate(table_rows, headers=["Class", "IoU", "F1", "Acc"], tablefmt="grid"))
#                 f.write("\n")
#                 f.write(tabulate(mean_rows, headers=mean_table.keys(), tablefmt="grid"))
#             self.metrics.reset()
#
#          elif "Polyp_Benchmark" in self.cfg["DATASET"]["name"]:
#              # --- (Keep Benchmark reporting as is, remember to reset metrics) ---
#              # CVC-300
#              sem_index = self.CVC_300_metrics.compute()
#              iou_strs = [f"{iou:.4f}" for iou in sem_index['IOUs']]
#              f1_strs = [f"{f1:.4f}" for f1 in sem_index['F1']]
#              acc_strs = [f"{acc:.4f}" for acc in sem_index['ACC']]
#              table_rows = list(zip(list(self.opt.class_list), iou_strs, f1_strs, acc_strs))
#              print("\n============== CVC_300 =================\n")
#              print(tabulate(table_rows, headers=["Class", "IoU", "F1", "Acc"], tablefmt="grid"))
#              self.CVC_300_metrics.reset()
#
#              # CVC-ClinicDB
#              sem_index = self.CVC_ClinicDB_metrics.compute()
#              iou_strs = [f"{iou:.4f}" for iou in sem_index['IOUs']]
#              f1_strs = [f"{f1:.4f}" for f1 in sem_index['F1']]
#              acc_strs = [f"{acc:.4f}" for acc in sem_index['ACC']]
#              table_rows = list(zip(list(self.opt.class_list), iou_strs, f1_strs, acc_strs))
#              print("\n============== CVC_ClinicDB =================\n")
#              print(tabulate(table_rows, headers=["Class", "IoU", "F1", "Acc"], tablefmt="grid"))
#              self.CVC_ClinicDB_metrics.reset()
#
#              # CVC-ColonDB
#              sem_index = self.CVC_ColonDB_metrics.compute()
#              iou_strs = [f"{iou:.4f}" for iou in sem_index['IOUs']]
#              f1_strs = [f"{f1:.4f}" for f1 in sem_index['F1']]
#              acc_strs = [f"{acc:.4f}" for acc in sem_index['ACC']]
#              table_rows = list(zip(list(self.opt.class_list), iou_strs, f1_strs, acc_strs))
#              print("\n============== CVC_ColonDB =================\n")
#              print(tabulate(table_rows, headers=["Class", "IoU", "F1", "Acc"], tablefmt="grid"))
#              self.CVC_ColonDB_metrics.reset()
#
#              # ETIS-LaribPolypDB
#              sem_index = self.ETIS_LaribPolypDB_metrics.compute()
#              iou_strs = [f"{iou:.4f}" for iou in sem_index['IOUs']]
#              f1_strs = [f"{f1:.4f}" for f1 in sem_index['F1']]
#              acc_strs = [f"{acc:.4f}" for acc in sem_index['ACC']]
#              table_rows = list(zip(list(self.opt.class_list), iou_strs, f1_strs, acc_strs))
#              print("\n============== ETIS_LaribPolypDB =================\n")
#              print(tabulate(table_rows, headers=["Class", "IoU", "F1", "Acc"], tablefmt="grid"))
#              self.ETIS_LaribPolypDB_metrics.reset()
#
#              # Kvasir
#              sem_index = self.Kvasir_metrics.compute()
#              iou_strs = [f"{iou:.4f}" for iou in sem_index['IOUs']]
#              f1_strs = [f"{f1:.4f}" for f1 in sem_index['F1']]
#              acc_strs = [f"{acc:.4f}" for acc in sem_index['ACC']]
#              table_rows = list(zip(list(self.opt.class_list), iou_strs, f1_strs, acc_strs))
#              print("\n============== Kvasir =================\n")
#              print(tabulate(table_rows, headers=["Class", "IoU", "F1", "Acc"], tablefmt="grid"))
#              self.Kvasir_metrics.reset()
#
#          else: # Non-benchmark, non-video
#              sem_index = self.metrics.compute()
#              iou_strs = [f"{iou:.4f}" for iou in sem_index['IOUs']]
#              f1_strs = [f"{f1:.4f}" for f1 in sem_index['F1']]
#              acc_strs = [f"{acc:.4f}" for acc in sem_index['ACC']]
#              table_rows = list(zip(list(self.opt.class_list), iou_strs, f1_strs, acc_strs))
#              print(f"\n============== {self.cfg['DATASET']['dataset']} =================\n")
#              print(tabulate(table_rows, headers=["Class", "IoU", "F1", "Acc"], tablefmt="grid"))
#              self.metrics.reset()
#
#     # Inside the seg_network class in models/build.py
#
#     # Inside the seg_network class in models/build.py
#
#     # Inside the seg_network class in models/build.py
#
#     # Inside the seg_network class in models/build.py
#
#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         """
#         Processes a batch (clip) of video data for prediction.
#         Expects batch items like 'low_freq', 'high_freq', 'label', 'img_paths'.
#         Returns predictions and info for all frames in the clip.
#         """
#         # === FIX: Use 'batch' instead of 'x' ===
#         if not isinstance(batch, dict) or not all(
#                 key in batch for key in ["low_freq", "high_freq", "label", "img_paths"]):
#             raise ValueError(
#                 "Batch must be a dictionary containing 'low_freq', 'high_freq', 'label', and 'img_paths'.")
#
#         # Extract shapes using 'batch'
#         T, C_in, H_in, W_in = batch["low_freq"].shape
#         _, _, H_lbl, W_lbl = batch["label"].shape  # Unpack T, C=1, H, W
#
#         # Add Batch Dimension before calling the model using 'batch'
#         low_freq_input = batch["low_freq"].unsqueeze(0)  # (1, T, C_in, H_in, W_in)
#         high_freq_input = batch["high_freq"].unsqueeze(0)  # (1, T, C_in, H_in, W_in)
#
#         out = self.model(low_freq_input, high_freq_input)  # (1, T, NumClasses, H_out, W_out)
#
#         # Remove Batch Dimension from output
#         out = out.squeeze(0)  # (T, NumClasses, H_out, W_out)
#
#         # Interpolate to match label size (H_lbl, W_lbl)
#         out = F.interpolate(out, size=(H_lbl, W_lbl), mode="bilinear", align_corners=False)
#         # Shape after interpolate: (T, NumClasses, H_lbl, W_lbl)
#
#         # Get final predictions (argmax) -> Shape: (T, H_lbl, W_lbl)
#         preds = out.softmax(dim=1).argmax(dim=1)  # dim=1 is Class dimension
#
#         # Extract image paths using 'batch'
#         img_paths_list = batch["img_paths"]  # List[T]
#
#         # Get original frequency maps using 'batch'
#         low_freq_flat = batch["low_freq"]  # (T, C_in, H_in, W_in)
#         high_freq_flat = batch["high_freq"]  # (T, C_in, H_in, W_in)
#         # === End FIX ===
#
#         return {
#             "preds": preds,  # Tensor shape (T, H_lbl, W_lbl)
#             "img_paths": img_paths_list,  # List of T strings
#             "low_freqs": low_freq_flat,  # Tensor shape (T, C_in, H_in, W_in)
#             "high_freqs": high_freq_flat  # Tensor shape (T, C_in, H_in, W_in)
#         }
#
#     # def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
#     #     x = batch
#     #     # print("Predict Step - Input keys:", x.keys()) # Add for debugging if needed
#     #
#     #     # --- MODIFICATION HERE ---
#     #     # Check the shape before unpacking
#     #     low_freq_shape = x["low_freq"].shape
#     #     # print(f"Predict Step - low_freq shape: {low_freq_shape}") # Debug print
#     #     if len(low_freq_shape) == 5:
#     #         # This case might occur if batch_size > 1 was somehow used for prediction
#     #         B, T, C, H_in, W_in = low_freq_shape
#     #     elif len(low_freq_shape) == 4:
#     #         # This is the expected case for batch_size=1 prediction
#     #         T, C, H_in, W_in = low_freq_shape
#     #         B = 1  # Implicit batch size is 1
#     #     else:
#     #         raise ValueError(f"Unexpected shape for low_freq: {low_freq_shape}")
#     #     # --- END MODIFICATION ---
#     #
#     #     # print(f"Predict Step - B={B}, T={T}, C={C}, H_in={H_in}, W_in={W_in}") # Debug print
#     #
#     #     # Run the model forward pass
#     #     out = self.model(x["low_freq"], x["high_freq"])  # Assuming model takes these inputs
#     #
#     #     # --- POTENTIAL MODIFICATION NEEDED HERE ---
#     #     # Check the output shape from the model
#     #     out_shape = out.shape
#     #     # print(f"Predict Step - model output shape: {out_shape}") # Debug print
#     #
#     #     # Assuming the model outputs shape (B, T, NumClasses, H_out, W_out) if B > 1
#     #     # OR (T, NumClasses, H_out, W_out) if B = 1 and model preserves T
#     #     # Adjust the unpacking based on the actual output shape
#     #     if len(out_shape) == 5 and B > 1:  # Should not happen with batch_size=1
#     #         _, _, C_out, H_out, W_out = out_shape
#     #         # If B=1 but model still outputs 5D tensor (1, T, C_out, H_out, W_out)
#     #         out = out.squeeze(0)  # Remove the batch dim for consistency below
#     #     elif len(out_shape) == 4 and B == 1:
#     #         # Assuming output is (T, C_out, H_out, W_out)
#     #         _, C_out, H_out, W_out = out_shape
#     #     else:
#     #         # Handle other potential output shapes or raise an error
#     #         raise ValueError(f"Unexpected model output shape: {out_shape} for input shape {low_freq_shape}")
#     #     # --- END POTENTIAL MODIFICATION ---
#     #
#     #     # --- Post-processing (Interpolation, Argmax) ---
#     #     # Interpolate to match input size H_in, W_in or original label size if needed
#     #     # Using H_in, W_in from the low_freq input for consistency
#     #     # Note: out is now expected to be (T, C_out, H_out, W_out)
#     #     out_interpolated = F.interpolate(
#     #         out, size=(H_in, W_in), mode="bilinear", align_corners=False
#     #     )  # Shape: (T, C_out, H_in, W_in)
#     #
#     #     # Get predictions by taking argmax along the class dimension (C_out)
#     #     preds = torch.argmax(out_interpolated, dim=1)  # Shape: (T, H_in, W_in)
#     #
#     #     # Prepare the return dictionary - Ensure keys match what pred.py expects
#     #     # We return results for the single clip (T frames) processed in this step
#     #     results = {
#     #         "preds": preds,  # Tensor [T, H, W]
#     #         "img_paths": x["img_paths"],  # List [T] of strings
#     #         "low_freqs": x["low_freq"].squeeze(0) if len(x["low_freq"].shape) == 5 else x["low_freq"],
#     #         # Tensor [T, C, H_in, W_in]
#     #         "high_freqs": x["high_freq"].squeeze(0) if len(x["high_freq"].shape) == 5 else x["high_freq"],
#     #         # Tensor [T, C, H_in, W_in]
#     #         # Add original labels if needed for comparison later, ensure label exists in batch
#     #         # "labels": x.get("label", None)       # Tensor [T, H, W] or None
#     #     }
#     #
#     #     # Optional: Check shapes of returned items
#     #     # print(f"Predict Step - Returning shapes: preds={results['preds'].shape}, "
#     #     #       f"low_freqs={results['low_freqs'].shape}, high_freqs={results['high_freqs'].shape}, "
#     #     #       f"img_paths len={len(results['img_paths'])}")
#     #
#     #     return results
#
#     # def predict_step(self, batch, batch_idx, dataloader_idx=0): # Add dataloader_idx for completeness
#     #     """
#     #     Processes a batch (clip) of video data for prediction.
#     #     Expects batch items like 'low_freq', 'high_freq', 'label', 'img_paths'.
#     #     Returns predictions and info for all frames in the clip.
#     #     """
#     #     if not isinstance(batch, dict) or not all(
#     #             key in batch for key in ["low_freq", "high_freq", "label", "img_paths"]):
#     #         raise ValueError(
#     #             "Batch must be a dictionary containing 'low_freq', 'high_freq', 'label', and 'img_paths'.")
#     #
#     #     x = batch
#     #     # Input shapes: low_freq/high_freq (B, T, C, H_in, W_in), label (B, T, H_lbl, W_lbl)
#     #     # B=1 for prediction using SlidingWindowClipSampler
#     #     B, T, C, H_in, W_in = x["low_freq"].shape
#     #     _, _, H_lbl, W_lbl = x["label"].shape
#     #
#     #     # === MODIFICATION ===
#     #     # Remove .unsqueeze(2) - WaveletNetPlus expects (B, T, C, H_in, W_in)
#     #     out = self.model(x["low_freq"], x["high_freq"])
#     #     # Model output shape is (B, T, NumClasses, H_out, W_out)
#     #
#     #     # Reshape and interpolate like in validation_step
#     #     num_classes = out.size(2)
#     #     out = out.view(B * T, num_classes, out.size(3), out.size(4))
#     #     out = F.interpolate(out, size=(H_lbl, W_lbl), mode="bilinear", align_corners=False)
#     #     # Shape after interpolate: (B*T, NumClasses, H_lbl, W_lbl)
#     #
#     #     # Get final predictions (argmax) -> Shape: (B*T, H_lbl, W_lbl)
#     #     preds = out.softmax(dim=1).argmax(dim=1)
#     #
#     #     # Since B=1, B*T = T. 'preds' shape is (T, H_lbl, W_lbl)
#     #
#     #     # Extract image paths: x["img_paths"] should be a list of T paths from the dataset __getitem__
#     #     img_paths_flat = x["img_paths"]
#     #     if B > 1 and isinstance(x["img_paths"][0], list): # Handle potential future B>1 case just in case
#     #          img_paths_flat = [p for sublist in x["img_paths"] for p in sublist]
#     #
#     #     # Reshape frequency maps to (B*T, C, H_in, W_in) -> (T, C, H_in, W_in)
#     #     low_freq_flat = x["low_freq"].view(B * T, C, H_in, W_in)
#     #     high_freq_flat = x["high_freq"].view(B * T, C, H_in, W_in)
#     #
#     #     # Return dictionary matching the structure expected by pred.py post-processing loop
#     #     return {
#     #         "preds": preds,              # Tensor shape (T, H_lbl, W_lbl)
#     #         "img_paths": img_paths_flat, # List of T strings
#     #         "low_freqs": low_freq_flat,  # Tensor shape (T, C, H_in, W_in)
#     #         "high_freqs": high_freq_flat # Tensor shape (T, C, H_in, W_in)
#     #     }
#
#     def configure_optimizers(self):
#         # --- (Keep this method as is) ---
#         # Filter parameters that require gradients
#         trainable_params = filter(lambda p: p.requires_grad, self.parameters())
#         # Print trainable parameter names (optional debug)
#         # print("Trainable parameters:")
#         # for name, param in self.model.named_parameters():
#         #      if param.requires_grad:
#         #          print(name)
#
#         optimizer = torch.optim.AdamW(
#             trainable_params, # Pass only trainable parameters
#             lr=self.cfg["OPTIMIZER"]["lr"],
#             betas=(0.9, 0.999),
#             weight_decay=self.cfg["OPTIMIZER"]["weight_decay"],
#         )
#
#         # Calculate total steps based on sampler length if available, else estimate
#         try:
#             # Assumes train_loader is accessible or total_samples is pre-calculated
#             # total_samples might be clips count here
#             total_steps = math.ceil(self.opt.total_samples / int(self.cfg["TRAIN"]["node"])) * self.cfg["TRAIN"]["nepoch"]
#             steps_per_epoch = math.ceil(self.opt.total_samples / int(self.cfg["TRAIN"]["node"]))
#             print(f"Total training steps calculated: {total_steps} (Epochs: {self.cfg['TRAIN']['nepoch']}, Steps/Epoch: {steps_per_epoch})")
#         except AttributeError:
#              # Fallback if self.opt.total_samples is not set correctly for clips
#              print("Warning: self.opt.total_samples not available, estimating scheduler steps.")
#              # Estimate steps per epoch based on dataset length and batch size (less accurate for samplers)
#              # This estimate might be wrong if total_samples refers to frames not clips.
#              # You might need to manually calculate total clips.
#              estimated_steps_per_epoch = 500 # Provide a reasonable guess or calculate properly
#              total_steps = estimated_steps_per_epoch * self.cfg["TRAIN"]["nepoch"]
#
#
#         # Using timm's CosineLRScheduler which is often robust
#         scheduler = CosineLRScheduler(
#              optimizer,
#              t_initial=total_steps, # Total number of steps
#              lr_min=1e-6, # Minimum learning rate
#              warmup_t=self.cfg["SCHEDULER"]["warmup_epoch"] * steps_per_epoch if 'steps_per_epoch' in locals() else self.cfg["SCHEDULER"]["warmup_epoch"] * 500, # Warmup steps
#              warmup_lr_init=self.cfg["SCHEDULER"]["lr_warmup"], # Initial warmup LR
#              cycle_limit=1, # Number of cycles
#              t_in_epochs=False, # Steps based scheduler
#         )
#
#         # Original scheduler (if preferred)
#         # scheduler = WarmupCosineAnnealingLR(
#         #     optimizer,
#         #     self.cfg["TRAIN"]["nepoch"],
#         #     self.cfg["SCHEDULER"]["warmup_epoch"],
#         #     steps_per_epoch if 'steps_per_epoch' in locals() else 500, # Steps per epoch needed
#         #     self.cfg["SCHEDULER"]["lr_warmup"],
#         #     self.cfg["SCHEDULER"]["warmup_ratio"],
#         # )
#
#
#         lr_scheduler_config = {
#             "scheduler": scheduler,
#             "interval": "step", # Scheduler steps every training step
#             "frequency": 1,
#         }
#
#         return dict(optimizer=optimizer, lr_scheduler=lr_scheduler_config)

