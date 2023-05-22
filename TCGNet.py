import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
from decoder import DECODER
from lib.models.cls_hrnet import HighResolutionNet
from capsules import PWV

# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

state_dict_path = './backbone/resnet/hrnetv2_w48_imagenet_pretrained.pth'

with open(
        r"/home/liuy/zl/TCGNet/TCGNet_hr/experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
        'r', encoding='utf-8') as f:
    temp = yaml.load(stream=f, Loader=yaml.FullLoader)

hrnet = HighResolutionNet(temp)
hrnet.load_state_dict(torch.load(state_dict_path))


class ConvBnRelu(nn.Module):
    # Dilationconvolution
    # batch normalization
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class mm(nn.Module):
    def __init__(self):
        super(mm, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        # self.bs1 = nn.Sequential(
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )
        # self.bs2 = nn.Sequential(
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )

    def forward(self, input1, input2):
        a = input1  # res
        b = input2  # acti

        c = torch.matmul(a, b.permute(0, 1, 3, 2))  # ww
        d = torch.matmul(a.permute(0, 1, 3, 2), b)  # hh
        # cnn
        e = torch.matmul(c, a)
        f = torch.matmul(e, d)
        f = self.bn1(f)
        out1 = f * a + a

        # capsule
        g = torch.matmul(c, b)
        h = torch.matmul(g, d)
        h = self.bn2(h)
        out2 = h * b + b

        return out1, out2


class UD(nn.Module):
    """  Upsample Concat Downsample  """

    def __init__(self):
        super(UD, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.cbr0 = ConvBnRelu(in_channels=256, out_channels=128)
        self.cbr1 = ConvBnRelu(in_channels=256, out_channels=128)
        self.cbr2 = ConvBnRelu(in_channels=256, out_channels=128)
        self.cbr3 = ConvBnRelu(in_channels=256, out_channels=128)

        self.downsample1 = ConvBnRelu(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.downsample2 = ConvBnRelu(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.downsample3 = ConvBnRelu(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.downsample4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1)
        )
        # self.outconv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, fea0, fea1, fea2, fea3):
        # out = self.cbr0(torch.cat([self.upsample(fea4), fea3], 1))  # 22
        out = self.cbr1(torch.cat([self.upsample(fea3), fea2], 1))  # 22
        out = self.cbr2(torch.cat([self.upsample(out), fea1], 1))  # 44
        out1 = self.cbr3(torch.cat([self.upsample(out), fea0], 1))  # 88
        out2 = self.downsample1(out1)  # 44
        out3 = self.downsample2(out2)  # 22
        # out4 = self.downsample3(out3)  # 11
        out4 = self.downsample4(out3)

        return out1, out2, out3, out4


class TCGNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(TCGNet, self).__init__()
        # params
        fea_dim = 128
        # capsnet
        self.capsnet = PWV(128)

        self.hr = hrnet

        # aspp + change channels
        # self.aspp4 = ASPP(in_channels=2048, atrous_rates=[6, 12, 18], out_channels=128)
        self.aspp3 = ASPP(in_channels=384, atrous_rates=[6, 12, 18], out_channels=128)
        self.aspp2 = ASPP(in_channels=192, atrous_rates=[6, 12, 18], out_channels=128)
        self.aspp1 = ASPP(in_channels=96, atrous_rates=[6, 12, 18], out_channels=128)
        self.aspp0 = ASPP(in_channels=48, atrous_rates=[6, 12, 18], out_channels=128)

        # upsample and downsample to get capsules-input
        self.ud = UD()

        self.decoder = DECODER()

        self.mm = mm()
        self.dropout = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        # layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        # layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        # layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        # layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        # layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]
        layer = self.hr(x)
        layer0 = layer[0]
        layer1 = layer[1]
        layer2 = layer[2]
        layer3 = layer[3]

        # channel reduction
        # cr4 = self.aspp4(layer4)  # 128
        cr3 = self.aspp3(layer3)  # 128
        cr2 = self.aspp2(layer2)  # 128
        cr1 = self.aspp1(layer1)  # 128
        cr0 = self.aspp0(layer0)  # 128
        # capsules-input and res-matrix
        out88, out44, out22, rm = self.ud(cr0, cr1, cr2, cr3)
        rm = self.dropout(rm)
        rm_sigmoid = torch.sigmoid(rm)

        # capsnet branch and matrix bmm
        activation, pose = self.capsnet(out88)
        activation = self.dropout(activation)
        activation_sigmoid = torch.sigmoid(activation)

        # Correlation matrix
        cnn_o, caps_o = self.mm(rm_sigmoid, activation_sigmoid)  # 1
        # cnn_o = self.dropout(cnn_o)
        cnn_o_sigmoid = torch.sigmoid(cnn_o)
        # cnn_o_relu = torch.relu(cnn_o)

        # caps_o = self.dropout(caps_o)
        caps_o_sigmoid = torch.sigmoid(caps_o)
        # caps_o_relu = torch.relu(caps_o)

        predict = self.decoder(cnn_o_sigmoid, caps_o_sigmoid, out22, out44, out88)

        if self.training:
            return predict

        return predict
