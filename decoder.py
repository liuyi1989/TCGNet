import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import yaml


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


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x1)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x2 * y.expand_as(x2)


class DECODER(nn.Module):
    def __init__(self):
        super(DECODER, self).__init__()
        self.cbr1 = ConvBnRelu(in_channels=129, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cbr2 = ConvBnRelu(in_channels=129, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cbr3 = ConvBnRelu(in_channels=129, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.cbr4 = ConvBnRelu(in_channels=129, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cbr5 = ConvBnRelu(in_channels=129, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cbr6 = ConvBnRelu(in_channels=129, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.cbr7 = ConvBnRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cbr8 = ConvBnRelu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.eca1 = eca_layer()
        self.eca2 = eca_layer()
        self.eca3 = eca_layer()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, cnn_in, caps_in, m1, m2, m3):
        """
        cnn_in : 22*22*1
        caps_in : 22*22*1
        m1 : 22*22*128
        m2 : 44*44*128
        m3 : 88*88*128
        """
        cnn_in_44 = F.interpolate(cnn_in, size=m2.size()[2:], mode='bilinear', align_corners=True)
        cnn_in_88 = F.interpolate(cnn_in, size=m3.size()[2:], mode='bilinear', align_corners=True)

        caps_in_44 = F.interpolate(caps_in, size=m2.size()[2:], mode='bilinear', align_corners=True)
        caps_in_88 = F.interpolate(caps_in, size=m3.size()[2:], mode='bilinear', align_corners=True)

        cnn1 = self.cbr1(torch.concat([cnn_in, m1], 1))
        cnn2 = self.cbr2(torch.concat([cnn_in_44, m2], 1))
        cnn3 = self.cbr3(torch.concat([cnn_in_88, m3], 1))

        caps1 = self.cbr4(torch.concat([caps_in, m1], 1))
        caps2 = self.cbr5(torch.concat([caps_in_44, m2], 1))
        caps3 = self.cbr6(torch.concat([caps_in_88, m3], 1))

        d1 = self.eca1(cnn1, caps1)
        d2 = self.eca2(cnn2, caps2)
        d3 = self.eca3(cnn3, caps3)

        out = self.cbr7(torch.concat([self.upsample(d1), d2], 1))
        out = self.cbr8(torch.concat([self.upsample(out), d3], 1))

        out = F.interpolate(out, size=(352, 352), mode='bilinear', align_corners=True)
        out = self.outconv(out)

        return out

