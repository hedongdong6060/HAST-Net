import torch
import random
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance, ImageChops
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import pywt


class Resize(object):
    def __init__(self, image_size, other_size):
        self.image_size = image_size  # size: (w, h)
        self.other_size = other_size

    def __call__(self, sample):
        low_freq = sample["low_freq"]
        high_freq = sample["high_freq"]
        mask = sample["label"]

        # 低频内容用双线性插值
        low_freq = cv2.resize(low_freq, self.other_size, interpolation=cv2.INTER_LINEAR)
        # 高频内容用Lanczos插值，更好地保留细节
        high_freq = cv2.resize(high_freq, self.other_size, interpolation=cv2.INTER_LANCZOS4)
        # 掩码用最近邻插值
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        return {"low_freq": low_freq, "high_freq": high_freq, "label": mask}


class EnhancedWaveletTransform(object):
    def __init__(self, wavelet="db2", level=1):
        self.wavelet = wavelet
        self.level = level

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        LL, (LH, HL, HH) = pywt.dwt2(img, self.wavelet)

        # 归一化各个分量
        LL = (LL - LL.min()) / (LL.max() - LL.min() + 1e-8)
        LH = (LH - LH.min()) / (LH.max() - LH.min() + 1e-8)
        HL = (HL - HL.min()) / (HL.max() - HL.min() + 1e-8)
        HH = (HH - HH.min()) / (HH.max() - HH.min() + 1e-8)

        # 加权融合高频分量
        # 边缘信息在HL和LH中更明显，给予更高权重
        merge1 = 0.5 * HH + 0.25 * HL + 0.25 * LH
        merge1 = (merge1 - merge1.min()) / (merge1.max() - merge1.min() + 1e-8)

        return {"low_freq": LL, "high_freq": merge1, "label": mask}


class MultiScaleWaveletTransform(object):
    def __init__(self, wavelet="db2", levels=2):
        self.wavelet = wavelet
        self.levels = levels

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 多尺度小波变换
        coeffs = pywt.wavedec2(img, self.wavelet, level=self.levels)

        # 提取低频
        LL = coeffs[0]
        LL = (LL - LL.min()) / (LL.max() - LL.min() + 1e-8)

        # 合并多尺度高频
        high_freq_components = []
        for level in range(1, len(coeffs)):
            LH, HL, HH = coeffs[level]
            LH = (LH - LH.min()) / (LH.max() - LH.min() + 1e-8)
            HL = (HL - HL.min()) / (HL.max() - HL.min() + 1e-8)
            HH = (HH - HH.min()) / (HH.max() - HH.min() + 1e-8)

            # 根据尺度赋予不同权重
            weight = 1.0 / (2 ** (level - 1))
            high_freq_components.append((LH + HL + HH) * weight)

        # 将不同尺度的高频分量调整到相同大小并融合
        merged_high_freq = np.zeros_like(high_freq_components[0])
        for comp in high_freq_components:
            if comp.shape != merged_high_freq.shape:
                comp = cv2.resize(comp, merged_high_freq.shape[::-1])
            merged_high_freq += comp

        merged_high_freq = (merged_high_freq - merged_high_freq.min()) / (
                    merged_high_freq.max() - merged_high_freq.min() + 1e-8)

        return {"low_freq": LL, "high_freq": merged_high_freq, "label": mask}


# class FrequencyAttentionTransform(object):
#     """结合频率注意力机制的小波变换"""
#
#     def __init__(self, wavelet="db2"):
#         self.wavelet = wavelet
#
#     def __call__(self, sample):
#         img = sample["image"]
#         mask = sample["label"]
#
#         if len(img.shape) == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # 小波分解
#         LL, (LH, HL, HH) = pywt.dwt2(img, self.wavelet)
#
#         # 计算高频分量的能量图
#         energy_LH = np.abs(LH)
#         energy_HL = np.abs(HL)
#         energy_HH = np.abs(HH)
#
#         # 归一化能量图
#         energy_LH = energy_LH / (np.max(energy_LH) + 1e-8)
#         energy_HL = energy_HL / (np.max(energy_HL) + 1e-8)
#         energy_HH = energy_HH / (np.max(energy_HH) + 1e-8)
#
#         # 计算注意力权重
#         total_energy = energy_LH + energy_HL + energy_HH + 1e-8
#         weight_LH = energy_LH / total_energy
#         weight_HL = energy_HL / total_energy
#         weight_HH = energy_HH / total_energy
#
#         # 加权融合高频分量
#         high_freq = weight_LH * LH + weight_HL * HL + weight_HH * HH
#
#         # 归一化处理
#         LL = (LL - LL.min()) / (LL.max() - LL.min() + 1e-8)
#         high_freq = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)
#
#         return {"low_freq": LL, "high_freq": high_freq, "label": mask}


class FrequencyAttentionTransform(object):
    """结合频率注意力机制的小波变换，支持彩色图像"""

    def __init__(self, wavelet="db2"):
        self.wavelet = wavelet

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        # 处理彩色图像
        if len(img.shape) == 3:
            # 分离通道
            channels = cv2.split(img)
            low_freq_channels = []
            high_freq_channels = []

            # 对每个通道分别进行小波变换
            for channel in channels:
                # 小波分解
                LL, (LH, HL, HH) = pywt.dwt2(channel, self.wavelet)

                # 计算高频分量的能量图
                energy_LH = np.abs(LH)
                energy_HL = np.abs(HL)
                energy_HH = np.abs(HH)

                # 归一化能量图
                energy_LH = energy_LH / (np.max(energy_LH) + 1e-8)
                energy_HL = energy_HL / (np.max(energy_HL) + 1e-8)
                energy_HH = energy_HH / (np.max(energy_HH) + 1e-8)

                # 计算注意力权重
                total_energy = energy_LH + energy_HL + energy_HH + 1e-8
                weight_LH = energy_LH / total_energy
                weight_HL = energy_HL / total_energy
                weight_HH = energy_HH / total_energy

                # 加权融合高频分量
                high_freq = weight_LH * LH + weight_HL * HL + weight_HH * HH

                # 归一化处理
                LL = (LL - LL.min()) / (LL.max() - LL.min() + 1e-8)
                high_freq = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)

                low_freq_channels.append(LL)
                high_freq_channels.append(high_freq)

            # 合并通道
            low_freq = np.stack(low_freq_channels, axis=2)
            high_freq = np.stack(high_freq_channels, axis=2)
        else:
            # 灰度图像处理（保留原来的逻辑）
            LL, (LH, HL, HH) = pywt.dwt2(img, self.wavelet)

            energy_LH = np.abs(LH)
            energy_HL = np.abs(HL)
            energy_HH = np.abs(HH)

            energy_LH = energy_LH / (np.max(energy_LH) + 1e-8)
            energy_HL = energy_HL / (np.max(energy_HL) + 1e-8)
            energy_HH = energy_HH / (np.max(energy_HH) + 1e-8)

            total_energy = energy_LH + energy_HL + energy_HH + 1e-8
            weight_LH = energy_LH / total_energy
            weight_HL = energy_HL / total_energy
            weight_HH = energy_HH / total_energy

            high_freq = weight_LH * LH + weight_HL * HL + weight_HH * HH

            LL = (LL - LL.min()) / (LL.max() - LL.min() + 1e-8)
            high_freq = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-8)

            low_freq = LL
            high_freq = high_freq

        return {"low_freq": low_freq, "high_freq": high_freq, "label": mask}


class Normalize_tensor(object):
    def __init__(self, Low_mean, Low_std, High_mean, High_std):
        self.Low_mean = np.array(Low_mean)
        self.Low_std = np.array(Low_std)
        self.High_mean = np.array(High_mean)
        self.High_std = np.array(High_std)

    def __call__(self, sample):
        low_freq = sample["low_freq"]
        high_freq = sample["high_freq"]
        mask = sample["label"]

        low_freq = np.array(low_freq)
        high_freq = np.array(high_freq)

        low_freq = low_freq.astype(np.float32)
        low_freq = low_freq - self.Low_mean
        low_freq = low_freq / self.Low_std

        high_freq = high_freq.astype(np.float32)
        high_freq = high_freq - self.High_mean
        high_freq = high_freq / self.High_std

        low_freq = torch.FloatTensor(low_freq)
        high_freq = torch.FloatTensor(high_freq)
        mask = torch.LongTensor(mask)

        return {"low_freq": low_freq, "high_freq": high_freq, "label": mask}


class ToTensor(object):
    def __call__(self, sample):
        low_freq = sample["low_freq"]
        high_freq = sample["high_freq"]
        mask = sample["label"]

        low_freq = np.array(low_freq)
        high_freq = np.array(high_freq)

        low_freq = torch.FloatTensor(low_freq)
        high_freq = torch.FloatTensor(high_freq)
        mask = torch.LongTensor(mask)

        return {"low_freq": low_freq, "high_freq": high_freq, "label": mask}







