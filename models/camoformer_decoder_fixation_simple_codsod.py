import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
        nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d,
        nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q = q * mask
            k = k * mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)


class MSA_head(nn.Module):
    def __init__(self, mode='dilation', dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)


class MSA_module(nn.Module):
    def __init__(self, dim=128):
        super(MSA_module, self).__init__()
        self.B_TA = MSA_head()
        self.F_TA = MSA_head()
        self.TA = MSA_head()
        self.Fuse = nn.Conv2d(3 * dim, dim, kernel_size=3, padding=1)
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x, side_x, mask):
        N, C, H, W = x.shape
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear')
        mask_d = mask.detach()
        mask_d = torch.sigmoid(mask_d)
        xf = self.F_TA(x, mask_d)
        xb = self.B_TA(x, 1 - mask_d)
        x = self.TA(x)
        x = torch.cat((xb, xf, x), 1)
        x = x.view(N, 3 * C, H, W)
        x = self.Fuse(x)
        D = self.Fuse2(side_x + side_x * x)
        return D

    def initialize(self):
        weight_init(self)


class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels * 3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels * 2)

        self.conv3 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()

        self.side_conv1 = nn.Conv2d(512, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(320, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        self.conv_block = Conv_Block(channels)

        self.fuse1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.MSA5 = MSA_module(dim=channels)
        self.MSA4 = MSA_module(dim=channels)
        self.MSA3 = MSA_module(dim=channels)
        self.MSA2 = MSA_module(dim=channels)

        # 分类部分
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.output_conv4 = nn.Conv2d(channels, 11, kernel_size=1)
        self.output_conv3 = nn.Conv2d(channels, 11, kernel_size=1)
        self.output_conv2 = nn.Conv2d(channels, 11, kernel_size=1)
        self.output_conv1 = nn.Conv2d(channels, 11, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def classify_feature(self, feat, conv, mask_gt):
        x = conv(feat)

        # 2. 扩展mask的通道数，从 [batch, 1, 88, 88] 到 [batch, 5, 88, 88]，以便与output匹配
        mask_gt = mask_gt.expand(-1, 11, -1, -1)  # 扩展通道数

        masked_logits = x * mask_gt  # [B, 6, H, W]
        masked_sum = masked_logits.sum(dim=[2, 3])  # [B, 6]，每类的 masked sum
        fixation_area = mask_gt.sum(dim=[2, 3]).clamp(min=1e-6)  # (batch, 1)
        target_logits = masked_sum / fixation_area  # [B, 6]，每类的 masked mean

        return x,target_logits

    def forward(self, E4, E3, E2, E1, mask):
        E4 = self.side_conv1(E4)
        E3 = self.side_conv2(E3)
        E2 = self.side_conv3(E2)
        E1 = self.side_conv4(E1)

        if E4.size()[2:] != E3.size()[2:]:
            E4 = F.interpolate(E4, size=E3.size()[2:], mode='bilinear', align_corners=True)
        if E2.size()[2:] != E3.size()[2:]:
            E2 = F.interpolate(E2, size=E3.size()[2:], mode='bilinear', align_corners=True)

        E5 = self.conv_block(E4, E3, E2)

        E4 = torch.cat((E4, E5), 1)
        E3 = torch.cat((E3, E5), 1)
        E2 = torch.cat((E2, E5), 1)

        E4 = F.relu(self.fuse1(E4), inplace=True)
        E3 = F.relu(self.fuse2(E3), inplace=True)
        E2 = F.relu(self.fuse3(E2), inplace=True)

        P5 = F.interpolate(mask, size=E5.size()[2:], mode='bilinear', align_corners=True)

        D4 = self.MSA5(E5, E4, P5)
        D4 = F.interpolate(D4, size=E3.size()[2:], mode='bilinear', align_corners=True)
        P4 = F.interpolate(mask, size=D4.size()[2:], mode='bilinear', align_corners=True)

        D3 = self.MSA4(D4, E3, P4)
        D3 = F.interpolate(D3, size=E2.size()[2:], mode='bilinear', align_corners=True)
        P3 = F.interpolate(mask, size=D3.size()[2:], mode='bilinear', align_corners=True)

        D2 = self.MSA3(D3, E2, P3)
        D2 = F.interpolate(D2, size=E1.size()[2:], mode='bilinear', align_corners=True)
        P2 = F.interpolate(mask, size=D2.size()[2:], mode='bilinear', align_corners=True)

        D1 = self.MSA2(D2, E1, P2)


        D4 = F.interpolate(D4, size=(mask.size(2), mask.size(3)), mode='bilinear', align_corners=True)
        D3 = F.interpolate(D3, size=(mask.size(2), mask.size(3)), mode='bilinear', align_corners=True)
        D2 = F.interpolate(D2, size=(mask.size(2), mask.size(3)), mode='bilinear', align_corners=True)
        D1 = F.interpolate(D1, size=(mask.size(2), mask.size(3)), mode='bilinear', align_corners=True)

        # 对每层输出进行图像级语义分类
        #out4,target_out4 = self.classify_feature(D4, self.output_conv4,mask)
        #out3,target_out3 = self.classify_feature(D3, self.output_conv3,mask)
        #out2,target_out2 = self.classify_feature(D2, self.output_conv2,mask)
        out4 = self.output_conv4(D4)
        out3 = self.output_conv3(D3)
        out2 = self.output_conv2(D2)
        out1,target_out1 = self.classify_feature(D1, self.output_conv1,mask)

        return out4,out3,out2,out1,target_out1


