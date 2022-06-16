"""
FPN特征金字塔
SSH多尺度加强感受野
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#---------------------------------------------------#
#   卷积块
#   Conv3x3 + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

#---------------------------------------------------#
#   卷积块
#   Conv1x1 + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


#---------------------------------------------------#
#   卷积块
#   Conv3x3 + BatchNormalization
#---------------------------------------------------#
def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

#---------------------------------------------------#
#   SSH多尺度加强感受野
#   1个3x3Conv 2个3x3Conv 3个3x3Conv  有padding,所以大小不变
#   最后通道为给定的out_channel
#---------------------------------------------------#
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1

        # 3x3卷积
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        # 利用两个3x3卷积替代5x5卷积
        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        # 利用三个3x3卷积替代7x7卷积
        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, inputs):
        conv3X3 = self.conv3X3(inputs)

        conv5X5_1 = self.conv5X5_1(inputs)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        # 注意这里的输入是conv5X5_1
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        # 所有结果堆叠起来
        # 拼接后通道数为 out_channel
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

#---------------------------------------------------#
#   FPN特征金字塔
#	下面注释是输入为 640,640,3的场景,输出宽高为 80, 40, 20
#	如果输入1280,1280,3时,最终宽高会翻倍,即160, 80, 40
#---------------------------------------------------#
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        #-------------------------------------------#
        #   对mobilenet0.25的三个特征层进行通道改变 1x1Conv
        #-------------------------------------------#
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        #-------------------------------------------#
        #   特征融合后的处理
        #-------------------------------------------#
        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, inputs):
        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #       C3  80, 80,  64
        #       C4  40, 40, 128
        #       C5  20, 20, 256
        #-------------------------------------------#
        inputs = list(inputs.values())

        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #       output1  80, 80,  64
        #       output2  40, 40, 128
        #       output3  20, 20, 256
        #-------------------------------------------#
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        #-------------------------------------------#
        #   output3上采样和output2特征融合
        #   output2  40, 40, 64
        #-------------------------------------------#
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        #-------------------------------------------#
        #   output2上采样和output1特征融合
        #   output1  80, 80, 64
        #-------------------------------------------#
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        # output1  80, 80, 64
        # output2  40, 40, 64
        # output3  20, 20, 64
        out = [output1, output2, output3]
        return out

