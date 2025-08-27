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


class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.Wxi = nn.Conv2d(
            input_channels, hidden_channels, kernel_size, 1, padding, bias=True
        )
        self.Whi = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size, 1, padding, bias=False
        )
        self.Wxf = nn.Conv2d(
            input_channels, hidden_channels, kernel_size, 1, padding, bias=True
        )
        self.Whf = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size, 1, padding, bias=False
        )
        self.Wxc = nn.Conv2d(
            input_channels, hidden_channels, kernel_size, 1, padding, bias=True
        )
        self.Whc = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size, 1, padding, bias=False
        )
        self.Wxo = nn.Conv2d(
            input_channels, hidden_channels, kernel_size, 1, padding, bias=True
        )
        self.Who = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size, 1, padding, bias=False
        )

    def forward(self, x, h, c):
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        g = torch.tanh(self.Wxc(x) + self.Whc(h))
        c_next = f * c + i * g
        o = torch.sigmoid(self.Wxo(x) + self.Who(h))
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):

        height, width = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width).cuda(),
            torch.zeros(batch_size, self.hidden_channels, height, width).cuda(),
        )


class WaveletNetPlus(nn.Module):

    def __init__(
        self, in_channels, num_classes, hidden_channels=128, num_diffusion_steps=4
    ):
        super(WaveletNetPlus, self).__init__()
        self.wavelet_net = WaveletNet(in_channels, num_classes, num_diffusion_steps)

        self.conv_lstm = ConvLSTMCell(
            input_channels=num_classes, hidden_channels=hidden_channels
        )

        self.output_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, input1_seq, input2_seq):

        B, T, C, H, W = input1_seq.shape
        h, c = self.conv_lstm.init_hidden(B, (H, W))

        outputs = []
        for t in range(T):
            x1_t = input1_seq[:, t]
            x2_t = input2_seq[:, t]


            out_t = self.wavelet_net(x1_t, x2_t)

            h, c = self.conv_lstm(out_t, h, c)

            out_t_final = self.output_conv(h)

            outputs.append(out_t_final.unsqueeze(1))

        return torch.cat(outputs, dim=1)


# 输入视频序列
#   ├─ input1_seq (B, T, C, H, W) ──┐
#   └─ input2_seq (B, T, C, H, W) ──┘
#             │
#             ▼
#       循环 t=0 到 T-1
#             │
#             ▼
#        提取当前帧
#   ├─ x1_t = input1_seq[:, t] ──┐
#   └─ x2_t = input2_seq[:, t] ──┘
#             │
#             ▼
#      WaveletNet(x1_t, x2_t)
#      - branch1 和 branch2 提取特征
#      - 融合 c1 到 c5
#      - WaveletDecoder 解码
#             │
#             ▼
#        out_t (初步分割图)
#             │
#             ▼
#      ConvLSTM(out_t, h_{t-1}, c_{t-1})
#      - 更新隐藏状态和细胞状态
#             │
#             ▼
#        h_t, c_t
#             │
#             ▼
#      output_conv(h_t)
#             │
#             ▼
#        out_t_final (最终分割图)
#             │
#             ▼
#       添加到 outputs 列表
#             │
#             ▼
#   拼接 outputs -> (B, T, num_classes, H, W)
#             │
#             ▼
#         输出视频序列的分割图
#
#
#
#
# 输入视频序列
#   ├─ input1_seq [B, T, C, H, W] ──┐
#   └─ input2_seq [B, T, C, H, W] ──┘
#             |
#             ▼
#       循环 t=0 到 T-1
#             |
#             ▼
#        提取当前帧
#   ├─ x1_t = input1_seq[:, t] [B, C, H, W] ──┐
#   └─ x2_t = input2_seq[:, t] [B, C, H, W] ──┘
#             |
#             ▼
#      WaveletNet(x1_t, x2_t)
#      |
#      ├── branch1(x1_t)
#      │   ├── conv layers → c1_b1 [64]
#      │   ├── down_conv → c2_b1 [128]
#      │   ├── down_conv → c3_b1 [256]
#      │   ├── down_conv → c4_b1 [512]
#      │   └── down_conv → c5_b1 [1024]
#      |
#      ├── branch2(x2_t)
#      │   ├── conv layers → c1_b2 [64]
#      │   ├── down_conv → c2_b2 [128]
#      │   ├── down_conv → c3_b2 [256]
#      │   ├── down_conv → c4_b2 [512]
#      │   └── down_conv → c5_b2 [1024]
#      |
#      └── 特征融合
#          ├── c1 = [c1_b1, c1_b2] [128]
#          ├── c2 = [c2_b1, c2_b2] [256]
#          ├── c3 = [c3_b1, c3_b2] [512]
#          ├── c4 = [c4_b1, c4_b2] [1024]
#          └── c5 = [c5_b1, c5_b2] [2048]
#             |
#             ▼
#      WaveletDecoder(c5, c4, c3, c2, c1)
#      |
#      ├── DiffusionBlock(diff5) → d5 [1024]
#      ├── up4(d5) → d4_up [512]
#      ├── Concat(d4_up, c4) → d4_in [1536]
#      ├── DiffusionBlock(diff4) → d4 [512]
#      ├── up3(d4) → d3_up [256]
#      ├── Concat(d3_up, c3) → d3_in [768]
#      ├── DiffusionBlock(diff3) → d3 [256]
#      ├── up2(d3) → d2_up [128]
#      ├── Concat(d2_up, c2) → d2_in [384]
#      ├── DiffusionBlock(diff2) → d2 [128]
#      ├── up1(d2) → d1_up [64]
#      ├── Concat(d1_up, c1) → d1_in [192]
#      ├── DiffusionBlock(diff1) → d1 [64]
#      └── out_conv(d1) → out_t [num_classes]
#             |
#             ▼
#      ConvLSTMCell(out_t, h_{t-1}, c_{t-1})
#      |
#      ├── 更新 h_t, c_t
#      |
#      ▼
#      output_conv(h_t) → out_t_final [num_classes]
#             |
#             ▼
#       添加到 outputs 列表
#             |
#             ▼
#   拼接 outputs → [B, T, num_classes, H, W]
#             |
#             ▼
#         输出视频序列的分割图




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class AttentionConvLSTMCell(nn.Module):
#     def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1, num_heads=4):
#         super(AttentionConvLSTMCell, self).__init__()
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.num_heads = num_heads
#
#         # QKV 卷积层
#         self.Wq = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding)
#         self.Wk = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding)
#         self.Wv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, 1, padding)
#
#         # 门控卷积层
#         self.Wxi = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, 1, padding)
#         self.Wxf = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, 1, padding)
#         self.Wxc = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, 1, padding)
#         self.Wxo = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, 1, padding)
#
#     def forward(self, x, h_prev, c_prev):
#         # 计算 QKV
#         q = self.Wq(h_prev)  # [B, hidden_channels, H, W]
#         k = self.Wk(h_prev)
#         v = self.Wv(h_prev)
#
#         # 多头注意力
#         B, C, H, W = q.shape
#         q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)  # [B, num_heads, H*W, C//num_heads]
#         k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
#         v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
#
#         attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (C // self.num_heads) ** 0.5
#         attn_weights = F.softmax(attn_weights, dim=-1)
#         attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, H*W, C//num_heads]
#         attn_output = attn_output.transpose(2, 3).contiguous().view(B, C, H, W)  # [B, hidden_channels, H, W]
#
#         # 融合输入和注意力输出
#         combined = torch.cat([x, attn_output], dim=1)  # [B, input_channels + hidden_channels, H, W]
#
#         # 门控计算
#         i = torch.sigmoid(self.Wxi(combined))  # 输入门
#         f = torch.sigmoid(self.Wxf(combined))  # 遗忘门
#         g = torch.tanh(self.Wxc(combined))     # 候选状态
#         c_next = f * c_prev + i * g            # 更新细胞状态
#         o = torch.sigmoid(self.Wxo(combined))  # 输出门
#         h_next = o * torch.tanh(c_next)        # 更新隐藏状态
#
#         return h_next, c_next
#
#     def init_hidden(self, batch_size, spatial_size):
#         height, width = spatial_size
#         return (
#             torch.zeros(batch_size, self.hidden_channels, height, width).cuda(),
#             torch.zeros(batch_size, self.hidden_channels, height, width).cuda(),
#         )
#
# # 更新 WaveletNetPlus
# class WaveletNetPlus(nn.Module):
#     def __init__(self, in_channels, num_classes, hidden_channels=128, num_diffusion_steps=4):
#         super(WaveletNetPlus, self).__init__()
#         self.wavelet_net = WaveletNet(in_channels, num_classes, num_diffusion_steps)
#         self.conv_lstm = AttentionConvLSTMCell(input_channels=num_classes, hidden_channels=hidden_channels)
#         self.output_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
#
#     def forward(self, input1_seq, input2_seq):
#         B, T, C, H, W = input1_seq.shape
#         h, c = self.conv_lstm.init_hidden(B, (H, W))
#         outputs = []
#
#         for t in range(T):
#             x1_t = input1_seq[:, t]
#             x2_t = input2_seq[:, t]
#             out_t = self.wavelet_net(x1_t, x2_t)
#             h, c = self.conv_lstm(out_t, h, c)
#             out_t_final = self.output_conv(h)
#             outputs.append(out_t_final.unsqueeze(1))
#
#         return torch.cat(outputs, dim=1)