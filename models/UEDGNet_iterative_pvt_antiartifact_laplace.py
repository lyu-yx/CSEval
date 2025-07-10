import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pvtv2 import pvt_v2_b2


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)
class SimpleDecoder(nn.Module):
    def __init__(self, channel):
        super(SimpleDecoder, self).__init__()

        # 逐步上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 融合特征
        self.fusion_conv = ConvBR(channel * 4, channel, 3, padding=1)

        self.output_conv1 = nn.Conv2d(channel, 2*channel, kernel_size=1)  # 输出5个类别
        self.output_conv2 = nn.Conv2d(2*channel, channel, kernel_size=1)  # 输出5个类别

        # 最终输出层，5个类别用于伪装等级评估
        self.output_conv = nn.Conv2d(channel, 5, kernel_size=1)  # 输出5个类别

        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 输出[batch, 5, 1, 1]

        # 修改全连接层的输入维度为 channel * 5
        self.fc = nn.Linear(5, 5)

        # Softmax层，用于计算类别概率
        self.softmax = nn.Softmax(dim=1)  # dim=1表示沿着类别维度应用Softmax

    def forward(self, f1, f2, f3, f4, mask):
        # 逐步上采样
        f2 = self.upsample(f2)  # H/8 -> H/4
        f3 = self.upsample(self.upsample(f3))  # H/16 -> H/4
        f4 = self.upsample(self.upsample(self.upsample(f4)))  # H/32 -> H/4

        # 拼接所有特征
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        # 融合
        fused = self.fusion_conv(fused)

        # 生成最终输出
        output1 = self.output_conv1(fused)
        output2 = self.output_conv2(output1)
        output = self.output_conv(output2)

        # 1. 将mask的维度从 [batch, 1, 352, 352] 缩放到 [batch, 1, 88, 88] （空间维度）
        mask = F.interpolate(mask, size=(output.size(2), output.size(3)), mode='bilinear', align_corners=True)

        # 2. 扩展mask的通道数，从 [batch, 1, 88, 88] 到 [batch, 5, 88, 88]，以便与output匹配
        mask = mask.expand(-1, 5, -1, -1)  # 扩展通道数

        # 3. 用mask对output做逐元素点乘（加权）
        output = output * mask  # 输出 [batch, 5, 88, 88]

        # 对输出进行全局平均池化，将空间维度压缩
        output = self.global_pool(output)  # 输出 [batch, 5, 1, 1]
        output = output.view(output.size(0), -1)  # 展平为 [batch, 5]

        # 通过全连接层得到最终的类别预测
        output = self.fc(output)  # 输出 [batch, 5]

        # 应用Softmax以获取类别概率
        output = self.softmax(output)  # 输出 [batch, 5]，每个值是类别的概率

        return output


class UEDGNet(nn.Module):
    def __init__(self, channel=32):
        super(UEDGNet, self).__init__()

        # Backbone (PVTv2)
        self.backbone = pvt_v2_b2()

        # 加载预训练权重
        path = './pre_trained/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # 特征维度缩减
        self.dr1 = DimensionalReduction(128, channel)  # 对应layer2输出
        self.dr2 = DimensionalReduction(320, channel)  # 对应layer3输出
        self.dr3 = DimensionalReduction(512, channel)  # 对应layer4输出

        # 简化解码器
        self.decoder = SimpleDecoder(channel)


    def forward(self, x, mask):
        # 提取特征
        pvt = self.backbone(x)
        fb1 = pvt[0]  # [batch, 64, H/4, W/4]
        fb2 = pvt[1]  # [batch, 128, H/8, W/8]
        fb3 = pvt[2]  # [batch, 320, H/16, W/16]
        fb4 = pvt[3]  # [batch, 512, H/32, W/32]

        # 维度缩减
        xr2 = self.dr1(fb2)
        xr3 = self.dr2(fb3)
        xr4 = self.dr3(fb4)

        # 解码器生成预测
        pred = self.decoder(fb1, xr2, xr3, xr4, mask)

        return pred


if __name__ == '__main__':
    net = UEDGNet(channel=64).eval()
    inputs = torch.randn(1, 3, 352, 352)
    mask = torch.randn(1, 1, 352, 352)
    output = net(inputs,mask)
    print(output)  # 应该输出: torch.Size([1, 5])
