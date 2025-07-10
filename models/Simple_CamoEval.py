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

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.fusion_conv = ConvBR(channel * 4, channel, 3, padding=1)
        self.fusion_conv1 = ConvBR(channel * 2, channel, 3, padding=1)
        self.fusion_conv2 = ConvBR(channel * 2, channel, 3, padding=1)
        self.fusion_conv3 = ConvBR(channel * 2, channel, 3, padding=1)

        self.output_conv1 = nn.Conv2d(channel, 2*channel, kernel_size=1)  
        self.output_conv2 = nn.Conv2d(2*channel, channel, kernel_size=1)  

        self.output_conv = nn.Conv2d(channel, 5, kernel_size=1)  

        self.global_pool = nn.AdaptiveAvgPool2d(1)  

        self.fc = nn.Linear(5, 5)

        self.softmax = nn.Softmax(dim=1)  

    def forward(self, f1, f2, f3, f4, mask):
        f2 = self.upsample(f2)  # H/8 -> H/4
        f3 = self.upsample(self.upsample(f3))  # H/16 -> H/4
        f4 = self.upsample(self.upsample(self.upsample(f4)))  # H/32 -> H/4

        fused1 = torch.cat([f4,f3], dim=1)
        fused1 = self.fusion_conv1(fused1)
        fused2 = torch.cat([fused1,f2], dim=1)
        fused2 = self.fusion_conv2(fused2)
        fused3 = torch.cat([fused2, f1], dim=1)
        fused3 = self.fusion_conv3(fused3)

        output1 = self.output_conv1(fused3)
        output2 = self.output_conv2(output1)
        output = self.output_conv(output2)

        mask = F.interpolate(mask, size=(output.size(2), output.size(3)), mode='bilinear', align_corners=True)

        mask = mask.expand(-1, 5, -1, -1)  

        output = output * mask  

        output = self.global_pool(output) 
        output = output.view(output.size(0), -1)  

        output = self.fc(output)  

        output = self.softmax(output) 

        return output

class UEDGNet(nn.Module):
    def __init__(self, channel=32):
        super(UEDGNet, self).__init__()

        # Backbone (PVTv2)
        self.backbone = pvt_v2_b2()

        path = './pre_trained/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.dr1 = DimensionalReduction(128, channel)  
        self.dr2 = DimensionalReduction(320, channel)  
        self.dr3 = DimensionalReduction(512, channel)  

        self.decoder = SimpleDecoder(channel)


    def forward(self, x, mask):
        pvt = self.backbone(x)
        fb1 = pvt[0]  # [batch, 64, H/4, W/4]
        fb2 = pvt[1]  # [batch, 128, H/8, W/8]
        fb3 = pvt[2]  # [batch, 320, H/16, W/16]
        fb4 = pvt[3]  # [batch, 512, H/32, W/32]

        xr2 = self.dr1(fb2)
        xr3 = self.dr2(fb3)
        xr4 = self.dr3(fb4)
        pred = self.decoder(fb1, xr2, xr3, xr4, mask)

        return pred


if __name__ == '__main__':
    net = UEDGNet(channel=64).eval()
    inputs = torch.randn(1, 3, 352, 352)
    mask = torch.randn(1, 1, 352, 352)
    output = net(inputs,mask)
    print(output)
