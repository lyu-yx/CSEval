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

        # 逐步上采�?
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 融合特征
        self.fusion_conv = ConvBR(channel * 4, channel, 3, padding=1)
        self.fusion_conv1 = ConvBR(channel * 2, channel, 3, padding=1)
        self.fusion_conv2 = ConvBR(channel * 2, channel, 3, padding=1)
        self.fusion_conv3 = ConvBR(channel * 2, channel, 3, padding=1)

        self.output_conv1 = nn.Conv2d(channel, 2*channel, kernel_size=3)  # 输出5个类�?
        self.output_conv2 = nn.Conv2d(2*channel, channel, kernel_size=3)  # 输出5个类�?
        self.output_conv = nn.Conv2d(channel, channel, kernel_size=3)  # 输出5个类�?

        self.output_conv_fixation = nn.Sequential(
            # 第一层卷�?(保持尺寸)
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # 第二层卷�?(保持尺寸)
            nn.Conv2d(channel, channel//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # 最终输出层
            nn.Conv2d(channel//2, 1, 1),  # 1x1卷积压缩到单通道
            nn.Sigmoid()  # 输出0-1
        )

        # 全局平均池化�?
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 输出[batch, 5, 1, 1]

        # 修改全连接层的输入维度为 channel * 5
        self.classify = nn.Conv2d(channel, 6, kernel_size=1)  # 输出5个类�?

        # Softmax层，用于计算类别概率
        self.softmax = nn.Softmax(dim=1)  # dim=1表示沿着类别维度应用Softmax

    def forward(self, f1, f2, f3, f4, mask):

        # 逐步上采�?
        f2 = self.upsample(f2)  # H/8 -> H/4
        f3 = self.upsample(self.upsample(f3))  # H/16 -> H/4
        f4 = self.upsample(self.upsample(self.upsample(f4)))  # H/32 -> H/4

        # 拼接所有特�?
        fused1 = torch.cat([f4,f3], dim=1)
        fused1 = self.fusion_conv1(fused1)
        fused2 = torch.cat([fused1,f2], dim=1)
        fused2 = self.fusion_conv2(fused2)
        fused3 = torch.cat([fused2, f1], dim=1)
        fused3 = self.fusion_conv3(fused3)

        # 生成最终输�?
        output1 = self.output_conv1(fused3)
        output2 = self.output_conv2(output1)
        logits = self.output_conv(output2)
        logits = F.interpolate(logits, size=(mask.size(2), mask.size(3)), mode='bilinear', align_corners=True)
        logits = self.classify(logits)  # 输出 [batch, 6,352,352]
        #sem = self.softmax(logits)  # 输出 [batch, 6,352,352]，每个值是类别的概�?
        fixation = self.output_conv_fixation(fused3)

        # 2. 扩展mask的通道数，�?[batch, 1, 352, 352] �?[batch, 6, 352, 352]，以便与output匹配
        mask = mask.expand(-1, 6, -1, -1)  # 扩展通道�?
        fixation = fixation.expand(-1, 6, -1, -1)  # 扩展通道�?

        # 3. 用mask对output做逐元素点乘（加权�?
        masked_logits = logits * mask  # 输出 [batch, 6, 352, 352]
        masked_logits = masked_logits * fixation
        masked_sum = masked_logits.sum(dim=[2, 3])  # (batch, 6)
        #mask_area = mask.sum(dim=[2, 3]).clamp(min=1e-6)  # (batch, 1)
        fixation_area = (fixation * mask).sum(dim=[2, 3]).clamp(min=1e-6)  # (batch, 1)
        target_logits = masked_sum / fixation_area  # (batch, 6)

        return logits, target_logits

class UEDGNet(nn.Module):
    def __init__(self, channel=64):
        super(UEDGNet, self).__init__()

        # Backbone (PVTv2)
        self.backbone = pvt_v2_b2()

        # 加载预训练权�?
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

        self.encoder_fusion = ConvBR(channel * 2, channel, 3, padding=1)

        # 简化解码器
        self.decoder = SimpleDecoder(channel)


    def forward(self, x, mask):
        fore = x * mask
        back = x * (1 - mask)
        # 提取特征
        pvt_fore = self.backbone(fore)
        pvt_back = self.backbone(back)
        fb1_f = pvt_fore[0]  # [batch, 64, H/4, W/4]
        fb2_f = pvt_fore[1]  # [batch, 128, H/8, W/8]
        fb3_f = pvt_fore[2]  # [batch, 320, H/16, W/16]
        fb4_f = pvt_fore[3]  # [batch, 512, H/32, W/32]

        fb1_b = pvt_back[0]  # [batch, 64, H/4, W/4]
        fb2_b = pvt_back[1]  # [batch, 128, H/8, W/8]
        fb3_b = pvt_back[2]  # [batch, 320, H/16, W/16]
        fb4_b = pvt_back[3]  # [batch, 512, H/32, W/32]

        # 维度缩减
        xr2_f = self.dr1(fb2_f)
        xr3_f = self.dr2(fb3_f)
        xr4_f = self.dr3(fb4_f)
        xr2_b = self.dr1(fb2_b)
        xr3_b = self.dr2(fb3_b)
        xr4_b = self.dr3(fb4_b)

        fb1 = torch.cat([fb1_f, fb1_b], dim=1)
        xr2 = torch.cat([xr2_f, xr2_b], dim=1)
        xr3 = torch.cat([xr3_f, xr3_b], dim=1)
        xr4 = torch.cat([xr4_f, xr4_b], dim=1)

        fb1 = self.encoder_fusion(fb1)
        xr2 = self.encoder_fusion(xr2)
        xr3 = self.encoder_fusion(xr3)
        xr4 = self.encoder_fusion(xr4)

        # 解码器生成预�?
        pred = self.decoder(fb1, xr2, xr3, xr4, mask)

        return pred


if __name__ == '__main__':
    net = UEDGNet(channel=64).eval()
    inputs = torch.randn(1, 3, 352, 352)
    mask = torch.randn(1, 1, 352, 352)
    sem, output = net(inputs,mask)
    print(sem[:,:,1,1])
    print(output)  # 应该输出: torch.Size([1, 5])
