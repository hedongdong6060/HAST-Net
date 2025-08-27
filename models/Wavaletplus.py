import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


def down_conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2, padding=0
    )


def same_conv(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )


def transition_conv(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out


class DoubleBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, transition=None):
        super(DoubleBasicBlock, self).__init__()
        self.block1 = BasicBlock(in_channels, out_channels)
        self.block2 = BasicBlock(out_channels, out_channels)
        self.transition = transition

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        if self.transition:
            out = self.transition(out)
        return out


# class DoubleBasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, transition=None):
#         super(DoubleBasicBlock, self).__init__()
#
#         # 简化结构：使用更小的中间通道压缩比
#         mid_channels = max(16, out_channels // 4)  # 更激进的压缩
#
#         # 合并第一个块，减少中间状态存储
#         self.main_path = nn.Sequential(
#             # 降维
#             nn.Conv2d(in_channels, mid_channels, 1, bias=False),
#             nn.BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),
#
#             # 深度可分离卷积
#             nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False),
#             nn.BatchNorm2d(mid_channels, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),
#
#             # 轻量级SE模块（减少通道降维度）
#             nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(mid_channels, max(4, mid_channels // 8), 1),  # 更激进的降维
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(max(4, mid_channels // 8), mid_channels, 1),
#                 nn.Sigmoid()
#             ) if in_channels > 64 else nn.Identity(),  # 小特征图才使用SE
#
#             # 恢复维度，合并到一个阶段减少中间存储
#             nn.Conv2d(mid_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True),  # 使用ReLU代替GELU节省内存
#
#             # 最后的深度卷积
#             nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
#             nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#         )
#
#         # 残差连接
#         self.shortcut = nn.Identity()
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                 nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#             )
#
#         self.relu = nn.ReLU(inplace=True)
#         self.transition = transition
#
#     def forward(self, x):
#         # 手动实现前向传播以优化内存
#         identity = self.shortcut(x)
#         out = self.main_path(x)
#         out = self.relu(out + identity)
#
#         if self.transition:
#             out = self.transition(out)
#
#         return out


# class DiffusionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, max_steps=4):
#         super(DiffusionBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.max_steps = max_steps
#         self.noise_layers = nn.ModuleList(
#             [nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1),
#                            nn.BatchNorm2d(in_channels),
#                            nn.ReLU(inplace=True)) for _ in range(max_steps)]
#         )
#         self.step_predictor = nn.Sequential(
#             nn.Conv2d(in_channels, 1, 1),  # 预测步数
#             nn.Sigmoid()  # 输出范围 [0, 1]
#         )
#         self.channel_project = nn.Conv2d(in_channels, out_channels, 1)
#
#     def forward(self, x):
#         step_score = self.step_predictor(x).mean(dim=(2, 3))  # [B, 1]
#         steps = (step_score * self.max_steps).round().long().clamp(1, self.max_steps)  # [B]
#         for b in range(x.size(0)):
#             for i in range(steps[b]):
#                 noise = torch.randn_like(x[b:b+1]) * 0.1
#                 x[b:b+1] = x[b:b+1] + noise
#                 x[b:b+1] = self.noise_layers[i](x[b:b+1])
#         x = self.channel_project(x)
#         return x



class DiffusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_diffusion_steps=4):
        super(DiffusionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_diffusion_steps)
            ]
        )
        self.channel_project = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        for layer in self.noise_layers:
            noise = torch.randn_like(x) * 0.1
            x = x + noise
            x = layer(x)
        x = self.channel_project(x)
        return x

# class ConditionalDiffusionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, num_steps=4):
#         super(ConditionalDiffusionBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # 强烈降低步骤数
#         self.num_steps = 1  # 最小化步骤数
#
#         # 极度简化的内容分析网络
#         self.content_analyzer = nn.Sequential(
#             nn.Conv2d(in_channels, 16, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 1, 1)
#         )
#
#         # 单个更轻量的噪声预测器
#         self.noise_predictor = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=max(1, in_channels // 16)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, 1)  # 使用1x1卷积替代第二个3x3卷积
#         )
#
#         # 固定的低噪声参数
#         self.noise_scale = 0.05
#
#         # 输出投影
#         self.out_proj = nn.Conv2d(in_channels, out_channels, 1)
#
#         # 残差连接
#         self.has_residual = (in_channels == out_channels)
#
#     def forward(self, x):
#         identity = x
#
#         # 如果在评估模式下，使用轻量版本
#         if not self.training:
#             output = self.out_proj(x)
#             if self.has_residual:
#                 output = output + identity
#             return output
#
#         # 简化的条件生成
#         condition_weight = torch.sigmoid(self.content_analyzer(x))
#
#         # 简化的单步扩散
#         if self.training:
#             with torch.no_grad():
#                 noise = torch.randn_like(x) * self.noise_scale
#
#             # 有条件地添加噪声
#             noised_input = x + noise * condition_weight
#
#             # 预测噪声残差
#             noise_residual = self.noise_predictor(noised_input)
#
#             # 更新当前特征
#             current = x + condition_weight * noise_residual
#
#             # 清理变量
#             del noised_input, noise_residual, noise
#         else:
#             current = x
#
#         # 输出投影
#         output = self.out_proj(current)
#
#         # 添加残差连接
#         if self.has_residual:
#             output = output + identity
#
#         return output


# 添加条件扩散块实现
# class ConditionalDiffusionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, num_steps=4):
#         super(ConditionalDiffusionBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # 可以考虑减少步骤数量
#         self.num_steps = min(num_steps, 3)  # 限制最大步骤数为3
#
#         # 减少中间特征通道数
#         mid_channels = min(64, in_channels)  # 限制最大中间通道
#         small_channels = min(32, mid_channels // 2)  # 进一步减少
#
#         # 简化内容分析网络
#         self.content_analyzer = nn.Sequential(
#             nn.Conv2d(in_channels, small_channels, 3, padding=1),
#             nn.InstanceNorm2d(small_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(small_channels, self.num_steps, 1)
#         )
#         self.softmax = nn.Softmax(dim=1)
#
#         # 减少噪声预测器的参数量
#         self.noise_predictors = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=min(8, in_channels)),  # 使用分组卷积
#                 nn.GroupNorm(min(8, in_channels), in_channels),
#                 nn.ReLU(inplace=True),  # 用ReLU替代SiLU以节省内存
#                 nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=min(8, in_channels)),
#                 nn.GroupNorm(min(8, in_channels), in_channels),
#                 nn.ReLU(inplace=True)
#             )
#             for _ in range(self.num_steps)
#         ])
#
#         # 降低噪声强度
#         self.noise_scales = nn.Parameter(torch.linspace(0.02, 0.1, self.num_steps).view(1, -1, 1, 1))
#
#         # 输出投影
#         self.out_proj = nn.Conv2d(in_channels, out_channels, 1)
#
#         # 残差连接
#         self.has_residual = (in_channels == out_channels)
#
#     def forward(self, x):
#         identity = x
#
#         # 只在训练模式下启用扩散过程，测试时简化
#         if not self.training and self.has_residual:
#             # 测试模式下简化计算
#             return self.out_proj(x) + identity
#
#         # 第1步：分析输入内容，生成空间条件图
#         condition_logits = self.content_analyzer(x)
#         condition_maps = self.softmax(condition_logits)  # [B, steps, H, W]
#
#         # 第2步：执行条件扩散过程
#         current = x
#
#         # 仅在训练时进行多步迭代，测试时减少步数
#         steps = self.num_steps if self.training else 1
#
#         for i in range(steps):
#             # 获取当前步骤的空间权重图
#             step_weight = condition_maps[:, i:i + 1]  # [B, 1, H, W]
#
#             # 使用torch.no_grad()生成噪声以节省内存
#             if self.training:
#                 with torch.no_grad():
#                     noise_level = self.noise_scales[:, i:i + 1]  # [1, 1, 1, 1]
#                     noise = torch.randn_like(current) * noise_level
#
#                 # 使用就地操作
#                 noised_input = current.clone()
#                 # 使用加法就地操作
#                 noised_input.add_(noise.mul(step_weight))
#
#                 # 预测去噪结果
#                 noise_residual = self.noise_predictors[i](noised_input)
#
#                 # 使用就地操作更新当前状态
#                 current.add_(step_weight.mul(noise_residual))
#
#                 # 主动释放中间变量
#                 del noised_input, noise_residual, noise
#             else:
#                 # 测试时简化计算流程
#                 noise_residual = self.noise_predictors[i](current)
#                 current = current + step_weight * noise_residual
#                 del noise_residual
#
#         # 第3步：输出投影
#         output = self.out_proj(current)
#
#         # 添加残差连接（如果输入输出通道数相同）
#         if self.has_residual:
#             output = output + identity
#
#         # 主动触发垃圾回收
#         import gc
#         gc.collect()
#
#         return output


class WaveletDecoder(nn.Module):
    def __init__(self, num_classes, num_diffusion_steps=4):
        super(WaveletDecoder, self).__init__()
        self.diff5 = DiffusionBlock(
            in_channels=2048, out_channels=1024, num_diffusion_steps=num_diffusion_steps
        )
        self.up4 = up_conv(1024, 512)

        self.diff4 = DiffusionBlock(
            in_channels=1536, out_channels=512, num_diffusion_steps=num_diffusion_steps
        )
        self.up3 = up_conv(512, 256)

        self.diff3 = DiffusionBlock(
            in_channels=768, out_channels=256, num_diffusion_steps=num_diffusion_steps
        )
        self.up2 = up_conv(256, 128)

        self.diff2 = DiffusionBlock(
            in_channels=384, out_channels=128, num_diffusion_steps=num_diffusion_steps
        )
        self.up1 = up_conv(128, 64)
        self.diff1 = DiffusionBlock(
            in_channels=192, out_channels=64, num_diffusion_steps=num_diffusion_steps
        )

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, c5, c4, c3, c2, c1):
        d5 = self.diff5(c5)
        d4_up = self.up4(d5)
        d4_in = torch.cat([d4_up, c4], dim=1)
        d4 = self.diff4(d4_in)

        d3_up = self.up3(d4)
        d3_in = torch.cat([d3_up, c3], dim=1)
        d3 = self.diff3(d3_in)

        d2_up = self.up2(d3)
        d2_in = torch.cat([d2_up, c2], dim=1)
        d2 = self.diff2(d2_in)

        d1_up = self.up1(d2)
        d1_in = torch.cat([d1_up, c1], dim=1)
        d1 = self.diff1(d1_in)

        out = self.out_conv(d1)
        return out
# class WaveletDecoder(nn.Module):
#     def __init__(self, num_classes, num_diffusion_steps=4):
#         super(WaveletDecoder, self).__init__()
#         # 使用条件扩散块替换原始扩散块
#         self.diff5 = ConditionalDiffusionBlock(
#             in_channels=2048, out_channels=1024, num_steps=num_diffusion_steps
#         )
#         self.up4 = up_conv(1024, 512)
#
#         self.diff4 = ConditionalDiffusionBlock(
#             in_channels=1536, out_channels=512, num_steps=num_diffusion_steps
#         )
#         self.up3 = up_conv(512, 256)
#
#         self.diff3 = ConditionalDiffusionBlock(
#             in_channels=768, out_channels=256, num_steps=num_diffusion_steps
#         )
#         self.up2 = up_conv(256, 128)
#
#         self.diff2 = ConditionalDiffusionBlock(
#             in_channels=384, out_channels=128, num_steps=num_diffusion_steps
#         )
#         self.up1 = up_conv(128, 64)
#         self.diff1 = ConditionalDiffusionBlock(
#             in_channels=192, out_channels=64, num_steps=num_diffusion_steps
#         )
#
#         self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
#
#     # forward方法保持不变
#     def forward(self, c5, c4, c3, c2, c1):
#         d5 = self.diff5(c5)
#         d4_up = self.up4(d5)
#         d4_in = torch.cat([d4_up, c4], dim=1)
#         d4 = self.diff4(d4_in)
#
#         d3_up = self.up3(d4)
#         d3_in = torch.cat([d3_up, c3], dim=1)
#         d3 = self.diff3(d3_in)
#
#         d2_up = self.up2(d3)
#         d2_in = torch.cat([d2_up, c2], dim=1)
#         d2 = self.diff2(d2_in)
#
#         d1_up = self.up1(d2)
#         d1_in = torch.cat([d1_up, c1], dim=1)
#         d1 = self.diff1(d1_in)
#
#         out = self.out_conv(d1)
#         return out


class WaveletNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_diffusion_steps=4):
        super(WaveletNet, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c), conv3x3(l1c, l1c), BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(
            l1c + l1c,
            l1c,
            nn.Sequential(
                conv1x1(l1c + l1c, l1c),
                nn.BatchNorm2d(l1c, momentum=BN_MOMENTUM),
            ),
        )

        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(
            l2c + l2c,
            l2c,
            nn.Sequential(
                conv1x1(l2c + l2c, l2c),
                nn.BatchNorm2d(l2c, momentum=BN_MOMENTUM),
            ),
        )
        self.b1_2_4_up = up_conv(l2c, l1c)

        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_3 = DoubleBasicBlock(
            l3c + l3c,
            l3c,
            nn.Sequential(
                conv1x1(l3c + l3c, l3c),
                nn.BatchNorm2d(l3c, momentum=BN_MOMENTUM),
            ),
        )
        self.b1_3_4_up = up_conv(l3c, l2c)

        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_3_down = down_conv(l4c, l4c)
        self.b1_4_3_same = same_conv(l4c, l4c)
        self.b1_4_4_transition = transition_conv(l4c + l5c + l4c, l4c)
        self.b1_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_6 = DoubleBasicBlock(
            l4c + l4c,
            l4c,
            nn.Sequential(
                conv1x1(l4c + l4c, l4c),
                nn.BatchNorm2d(l4c, momentum=BN_MOMENTUM),
            ),
        )
        self.b1_4_7_up = up_conv(l4c, l3c)

        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_2_up = up_conv(l5c, l5c)
        self.b1_5_2_same = same_conv(l5c, l5c)
        self.b1_5_3_transition = transition_conv(l5c + l5c + l4c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # branch2
        self.b2_1_1 = nn.Sequential(
            conv3x3(1, l1c), conv3x3(l1c, l1c), BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(
            l1c + l1c,
            l1c,
            nn.Sequential(
                conv1x1(l1c + l1c, l1c),
                nn.BatchNorm2d(l1c, momentum=BN_MOMENTUM),
            ),
        )

        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(
            l2c + l2c,
            l2c,
            nn.Sequential(
                conv1x1(l2c + l2c, l2c),
                nn.BatchNorm2d(l2c, momentum=BN_MOMENTUM),
            ),
        )
        self.b2_2_4_up = up_conv(l2c, l1c)

        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_3 = DoubleBasicBlock(
            l3c + l3c,
            l3c,
            nn.Sequential(
                conv1x1(l3c + l3c, l3c),
                nn.BatchNorm2d(l3c, momentum=BN_MOMENTUM),
            ),
        )
        self.b2_3_4_up = up_conv(l3c, l2c)

        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_2_down = down_conv(l4c, l5c)
        self.b2_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_3_down = down_conv(l4c, l4c)
        self.b2_4_3_same = same_conv(l4c, l4c)
        self.b2_4_4_transition = transition_conv(l4c + l5c + l4c, l4c)
        self.b2_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_6 = DoubleBasicBlock(
            l4c + l4c,
            l4c,
            nn.Sequential(
                conv1x1(l4c + l4c, l4c),
                nn.BatchNorm2d(l4c, momentum=BN_MOMENTUM),
            ),
        )
        self.b2_4_7_up = up_conv(l4c, l3c)

        self.b2_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_2_up = up_conv(l5c, l5c)
        self.b2_5_2_same = same_conv(l5c, l5c)
        self.b2_5_3_transition = transition_conv(l5c + l5c + l4c, l5c)
        self.b2_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_5_up = up_conv(l5c, l4c)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.decoder = WaveletDecoder(num_classes, num_diffusion_steps)

    def forward(self, input1, input2):
        x1_1 = self.b1_1_1(input1)
        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)
        x1_3 = self.b1_2_2_down(x1_2)
        x1_3 = self.b1_3_1(x1_3)
        x1_4_1 = self.b1_3_2_down(x1_3)
        x1_4_1 = self.b1_4_1(x1_4_1)
        x1_4_2 = self.b1_4_2_down(x1_4_1)
        x1_4_2 = self.b1_5_1(x1_4_2)

        x2_1 = self.b2_1_1(input2)
        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)
        x2_3 = self.b2_2_2_down(x2_2)
        x2_3 = self.b2_3_1(x2_3)
        x2_4_1 = self.b2_3_2_down(x2_3)
        x2_4_1 = self.b2_4_1(x2_4_1)
        x2_4_2 = self.b2_4_2_down(x2_4_1)
        x2_4_2 = self.b2_5_1(x2_4_2)

        c5 = torch.cat([x1_4_2, x2_4_2], dim=1)
        c4 = torch.cat([x1_4_1, x2_4_1], dim=1)
        c3 = torch.cat([x1_3, x2_3], dim=1)
        c2 = torch.cat([x1_2, x2_2], dim=1)
        c1 = torch.cat([x1_1, x2_1], dim=1)

        out = self.decoder(c5, c4, c3, c2, c1)
        return out



import torch.nn.functional as F
# # 动态注意力融合模块
class DynamicAttentionFusion(nn.Module):
    def __init__(self, channels):
        super(DynamicAttentionFusion, self).__init__()
        # 确保最小通道数不会为0
        reduction_factor = 8 if channels >= 16 else 2
        reduced_channels = max(1, channels // reduction_factor)

        self.query_conv = nn.Conv2d(channels, reduced_channels, 1)  # 查询卷积
        self.key_conv = nn.Conv2d(channels, reduced_channels, 1)  # 键卷积
        self.value_conv = nn.Conv2d(channels, channels, 1)  # 值卷积
        self.gamma = nn.Parameter(torch.zeros(1))  # 融合强度参数
        self.reduced_channels = reduced_channels

    def forward(self, current_feat, history_feats):
        # current_feat: 当前帧特征 [B, C, H, W]
        # history_feats: 历史帧特征列表 [B, C, H, W] 的列表
        B, C, H, W = current_feat.shape
        k = len(history_feats)

        # 简化的注意力机制
        query = self.query_conv(current_feat)  # [B, reduced_C, H, W]
        keys = [self.key_conv(h) for h in history_feats]  # 列表of [B, reduced_C, H, W]
        values = [self.value_conv(h) for h in history_feats]  # 列表of [B, C, H, W]

        # 空间注意力 - 对每个历史帧计算与当前帧的相似度
        attention_weights = []
        for i in range(k):
            # 计算点积相似度
            similarity = query * keys[i]  # 逐元素乘法 [B, reduced_C, H, W]
            similarity = similarity.sum(dim=1, keepdim=True)  # [B, 1, H, W]
            attention_weights.append(similarity)

        # 将注意力权重堆叠并应用softmax
        attention_weights = torch.cat(attention_weights, dim=1)  # [B, k, H, W]
        attention_weights = F.softmax(attention_weights, dim=1)  # 在历史帧维度上进行softmax

        # 使用注意力权重加权融合历史帧特征
        fused_history = torch.zeros_like(current_feat)
        for i in range(k):
            weight = attention_weights[:, i:i + 1, :, :]  # [B, 1, H, W]
            weight = weight.expand_as(values[i])  # [B, C, H, W]
            fused_history += weight * values[i]

        # 融合当前帧和历史帧
        return current_feat + self.gamma * fused_history


#
# # 修改后的 WaveletNetPlus


class ConvSLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvSLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # 输入门（指数门控）
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, padding, bias=True)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding, bias=False)

        # 遗忘门
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, padding, bias=True)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding, bias=False)

        # 细胞状态更新
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, padding, bias=True)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding, bias=False)

        # 输出门
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, 1, padding, bias=True)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding, bias=False)

    def forward(self, x, h, c):
        # 输入门使用指数门控
        i = torch.exp(self.Wxi(x) + self.Whi(h))  # sLSTM 的创新点
        # 遗忘门
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        # 细胞状态更新
        g = torch.tanh(self.Wxc(x) + self.Whc(h))
        c_next = f * c + i * g
        # 输出门
        # o = torch.exp(self.Wxo(x) + self.Who(h))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h))
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width).cuda(),
            torch.zeros(batch_size, self.hidden_channels, height, width).cuda(),
        )
#
#
# # 更新 WaveletNetPlus
# class WaveletNetPlus(nn.Module):
#     def __init__(self, in_channels, num_classes, hidden_channels=128, num_history=3):
#         super(WaveletNetPlus, self).__init__()
#         self.wavelet_net = WaveletNet(in_channels, num_classes)
#         self.attention_fusion = DynamicAttentionFusion(num_classes)
#         self.conv_lstm = ConvSLSTMCell(input_channels=num_classes, hidden_channels=hidden_channels)
#         self.output_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
#         self.num_history = num_history
#
#     def forward(self, input1_seq, input2_seq):
#         B, T, C, H, W = input1_seq.shape
#         h, c = self.conv_lstm.init_hidden(B, (H, W))
#         outputs = []
#
#         # 处理前num_history帧（无历史信息）
#         for t in range(self.num_history):
#             current_frame = (input1_seq[:, t], input2_seq[:, t])
#             current_feat = self.wavelet_net(*current_frame)
#             # 没有历史帧可融合，直接输出当前帧结果
#             h, c = self.conv_lstm(current_feat, h, c)
#             out_t = self.output_conv(h)
#             outputs.append(out_t.unsqueeze(1))
#
#         # 处理剩余帧（有历史信息）
#         for t in range(self.num_history, T):
#             current_frame = (input1_seq[:, t], input2_seq[:, t])
#             history_frames = [(input1_seq[:, t - i], input2_seq[:, t - i]) for i in range(1, self.num_history + 1)]
#             current_feat = self.wavelet_net(*current_frame)
#             history_feats = [self.wavelet_net(*hf) for hf in history_frames]
#             fused_feat = self.attention_fusion(current_feat, history_feats)
#             h, c = self.conv_lstm(fused_feat, h, c)
#             out_t = self.output_conv(h)
#             outputs.append(out_t.unsqueeze(1))
#
#         return torch.cat(outputs, dim=1)  # 返回所有T帧的输出


class WaveletNetPlus(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=128, num_history=3):
        super(WaveletNetPlus, self).__init__()
        self.wavelet_net = WaveletNet(in_channels, num_classes)
        self.attention_fusion = DynamicAttentionFusion(num_classes)
        self.conv_lstm = ConvSLSTMCell(input_channels=num_classes, hidden_channels=hidden_channels)
        self.output_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        self.num_history = num_history

    # def forward(self, input1_seq, input2_seq):
    #     B, T, C, H, W = input1_seq.shape
    #     h, c = self.conv_lstm.init_hidden(B, (H, W))
    #     outputs = []
    #
    #     # 缓存已计算的特征
    #     feature_cache = []
    #
    #     for t in range(T):
    #         current_frame = (input1_seq[:, t], input2_seq[:, t])
    #
    #         # 计算当前帧特征并缓存
    #         current_feat = self.wavelet_net(*current_frame)
    #         feature_cache.append(current_feat)
    #
    #         if t > 0:
    #             # 使用缓存的历史特征
    #             available_history = min(t, self.num_history)
    #             history_feats = feature_cache[max(0, t - available_history):t]
    #
    #             if history_feats:
    #                 current_feat = self.attention_fusion(current_feat, history_feats)
    #
    #         h, c = self.conv_lstm(current_feat, h, c)
    #         out_t = self.output_conv(h)
    #         outputs.append(out_t.unsqueeze(1))
    #
    #         # 如果内存仍然不足，可以限制缓存大小
    #         if len(feature_cache) > self.num_history + 1:
    #             feature_cache.pop(0)  # 移除最旧的特征
    #
    #     return torch.cat(outputs, dim=1)

    # def forward(self, input1_seq, input2_seq):
    #     B, T, C, H, W = input1_seq.shape
    #     h, c = self.conv_lstm.init_hidden(B, (H, W))
    #     outputs = []
    #
    #     # 处理所有帧
    #     for t in range(T):
    #         current_frame = (input1_seq[:, t], input2_seq[:, t])
    #         current_feat = self.wavelet_net(*current_frame)
    #
    #         # 对于有历史帧的情况，使用注意力融合
    #         if t > 0:
    #             # 使用所有可用的历史帧，但不超过num_history帧
    #             available_history = min(t, self.num_history)
    #             history_frames = [(input1_seq[:, t - i], input2_seq[:, t - i])
    #                               for i in range(1, available_history + 1)]
    #             history_feats = [self.wavelet_net(*hf) for hf in history_frames]
    #
    #             # 当有历史帧时才进行融合
    #             if history_feats:
    #                 current_feat = self.attention_fusion(current_feat, history_feats)
    #
    #         h, c = self.conv_lstm(current_feat, h, c)
    #         out_t = self.output_conv(h)
    #         outputs.append(out_t.unsqueeze(1))
    #
    #     return torch.cat(outputs, dim=1)

    def forward(self, input1_seq, input2_seq):
        B, T, C, H, W = input1_seq.shape
        h, c = self.conv_lstm.init_hidden(B, (H, W))
        outputs = []

        # 缓存特征，避免重复计算
        feature_cache = []

        for t in range(T):
            current_frame = (input1_seq[:, t], input2_seq[:, t])
            current_feat = self.wavelet_net(*current_frame)
            feature_cache.append(current_feat)

            if t > 0:
                # 使用缓存的特征，不重新计算
                available_history = min(t, self.num_history)
                history_feats = feature_cache[max(0, t - available_history):t]

                if history_feats:
                    current_feat = self.attention_fusion(current_feat, history_feats)

            h, c = self.conv_lstm(current_feat, h, c)
            out_t = self.output_conv(h)
            outputs.append(out_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)


