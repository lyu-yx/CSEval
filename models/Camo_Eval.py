import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pvtv2 import pvt_v2_b2
from models.camoformer_decoder import Decoder


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


class UEDGNet(nn.Module):
    def __init__(self, channel=32):
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

        # 简化解码器
        self.decoder = Decoder(128)


    def forward(self, x, mask):
        # 提取特征
        pvt = self.backbone(x)
        fb1 = pvt[0]  # [batch, 64, H/4, W/4]
        fb2 = pvt[1]  # [batch, 128, H/8, W/8]
        fb3 = pvt[2]  # [batch, 320, H/16, W/16]
        fb4 = pvt[3]  # [batch, 512, H/32, W/32]

        # 解码器生成预�?
        #pred4,pred3,pred2,pred = self.decoder(fb4, fb3, fb2, fb1, mask)
        pred,target_pred = self.decoder(fb4, fb3, fb2, fb1, mask)

        return pred,target_pred
        #return pred


if __name__ == '__main__':
    net = UEDGNet(channel=64).eval()
    inputs = torch.randn(1, 3, 352, 352)
    mask = torch.randn(1, 1, 352, 352)
    sem,output = net(inputs,mask)
    print(sem[:, :, 1, 1])
    print(output)  # 应该输出: torch.Size([1, 5])
