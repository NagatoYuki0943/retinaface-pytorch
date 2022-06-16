#---------------------------------------------------#
#   MobileNetV1-0.25
#   宽度为原版的0.25倍
#---------------------------------------------------#

import torch.nn as nn

#---------------------------------------------------#
#   Conv+BN+LeakyReLU
#---------------------------------------------------#
def conv_bn(inp, oup, stride = 1, leaky = 0.1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

#---------------------------------------------------#
#   dw+pw
#   3x3DWConv + 1x1Conv
#---------------------------------------------------#
def conv_dw(inp, oup, stride = 1, leaky=0.1):
    return nn.Sequential(
        # 3x3Conv
        # in = put = groups
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        # 1x1Conv
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

#---------------------------------------------------#
#   MobileNetV1-0.25
#	下面注释是输入为 640,640,3的场景,输出宽高为 80, 40, 20
#	如果输入1280,1280,3时,最终宽高会翻倍,即160, 80, 40
#---------------------------------------------------#
class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 640,640,3 -> 320,320,8
            conv_bn(3, 8, 2, leaky = 0.1),
            # 320,320,8 -> 320,320,16
            conv_dw(8, 16, 1),

            # 320,320,16 -> 160,160,32
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),

            # 160,160,32 -> 80,80,64
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
        )
        # 80,80,64 -> 40,40,128
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
        )
        # 40,40,128 -> 20,20,256
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        # 在这里打印都没效果
        # print(x.size())
        x = self.stage1(x)  # 640,640,3 -> 80,80, 64
        # print(x.size())
        x = self.stage2(x)  # 80,80, 64 -> 40,40,128
        # print(x.size())
        x = self.stage3(x)  # 40,40,128 -> 20,20,256
        # print(x.size())
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
