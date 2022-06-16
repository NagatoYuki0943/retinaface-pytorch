"""
下面注释是输入为 640,640,3的场景,输出宽高为 80, 40, 20
如果输入1280,1280,3时,最终宽高会翻倍,即160, 80, 40
num_anchors=2 代表每个特征点有两个先验框
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torchvision import models

from nets.layers import FPN, SSH
from nets.mobilenet025 import MobileNetV1


#---------------------------------------------------#
#   种类预测（是否包含人脸） 1x1Conv调整通道
#   输出通道为2:
#       out[0] > out[1]: 没有人脸
#       out[0] < out[1]: 有人脸
#---------------------------------------------------#
class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super().__init__()
        self.num_anchors = num_anchors
        #---------------------------------------------------#
        #   b, 64, 20, 20 -> b, 4, 20, 20
        #   b, 64, 40, 40 -> b, 4, 40, 40
        #   b, 64, 80, 80 -> b, 4, 80, 80
        #---------------------------------------------------#
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        #---------------------------------------------------#
        #   b, 所有框, 是否有人脸(1大有人脸,0大没人脸)
        #   b, 4, 20, 20 -> b, 20, 20, 4 -> b, 20*20*2, 2
        #   b, 4, 40, 40 -> b, 40, 40, 4 -> b, 40*40*2, 2
        #   b, 4, 80, 80 -> b, 80, 80, 4 -> b, 80*80*2, 2
        #---------------------------------------------------#
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)

#---------------------------------------------------#
#   预测框预测  1x1Conv调整通道
#   输出通道为4: 中心宽高调整
#---------------------------------------------------#
class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super().__init__()
        #---------------------------------------------------#
        #   b, 64, 20, 20 -> b, 8, 20, 20
        #   b, 64, 40, 40 -> b, 8, 40, 40
        #   b, 64, 80, 80 -> b, 8, 80, 80
        #---------------------------------------------------#
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        #---------------------------------------------------#
        #   b, 所有框, 中心宽高
        #   b, 8, 20, 20 -> b, 20, 20, 8 -> b, 20*20*2, 4
        #   b, 8, 40, 40 -> b, 40, 40, 8 -> b, 40*40*2, 4
        #   b, 8, 80, 80 -> b, 80, 80, 8 -> b, 80*80*2, 4
        #---------------------------------------------------#
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)

#---------------------------------------------------#
#   人脸关键点预测  1x1Conv调整通道
#   5: 5个人脸关键点
#   2: 5个关键点的调整参数
#---------------------------------------------------#
class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super().__init__()
        #---------------------------------------------------#
        #   b, 64, 20, 20 -> b, 20, 20, 20
        #   b, 64, 40, 40 -> b, 20, 40, 40
        #   b, 64, 80, 80 -> b, 20, 80, 80
        #---------------------------------------------------#
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        #---------------------------------------------------#
        #   转换为 b,所有框,5个人脸关键点和2个调整参数
        #   b, 20, 20, 20 -> b, 20, 20, 20 -> b, 20*20*2, 10
        #   b, 20, 40, 40 -> b, 40, 40, 20 -> b, 40*40*2, 10
        #   b, 20, 80, 80 -> b, 80, 80, 20 -> b, 80*80*2, 10
        #---------------------------------------------------#
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)

#-------------------------------------------#
#   1. 主干提取三层特征
#   2. FPN特征金字塔
#   3. SSH多尺度加强感受野,调整后宽高不变
#   4. ClassHead
#       b,所有框,2(有没有人脸)
#   4. BboxHead
#       b,所有框,4(框的调整)
#   4. LandmarkHead
#       b,所有框,10(5个特征点和位移)
#-------------------------------------------#
class RetinaFace(nn.Module):
    def __init__(self, cfg = None, pretrained = False, mode = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super().__init__()
        self.mode = mode
        backbone = None
        #-------------------------------------------#
        #   选择使用mobilenet0.25、resnet50作为主干
        #-------------------------------------------#
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if pretrained:
                checkpoint = torch.load("./model_data/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=pretrained)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])  # {cfg['return_layers'] = 'stage1': 1, 'stage2': 2, 'stage3': 3}

        #-------------------------------------------#
        #   获得每个初步有效特征层的通道数
        #-------------------------------------------#
        in_channels_list = [
            cfg['in_channel'] * 2,  # 'in_channel' : 32
            cfg['in_channel'] * 4,
            cfg['in_channel'] * 8
            ]
        out_channels = cfg['out_channel']

        #-------------------------------------------#
        #   利用初步有效特征层构建特征金字塔
        #-------------------------------------------#
        self.fpn = FPN(in_channels_list, out_channels)    # 'out_channel' : 64

        #-------------------------------------------#
        #   利用ssh模块提高模型感受野
        #-------------------------------------------#
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        #-------------------------------------------#
        #   依次创建3个ClassHead,BboxHead,LandmarkHead
        #-------------------------------------------#
        self.ClassHead    = self._make_class_head(   fpn_num=3, inchannels=out_channels)
        self.BboxHead     = self._make_bbox_head(    fpn_num=3, inchannels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channels)

    #-------------------------------------------#
    #   依次创建3个ClassHead,BboxHead,LandmarkHead
    #-------------------------------------------#
    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self,fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self,inputs):
        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #       C3  b, 64 , 80, 80
        #       C4  b, 128, 40, 40
        #       C5  b, 256, 20, 20
        #-------------------------------------------#
        out = self.body(inputs)

        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #       output1  b, 64, 80, 80
        #       output2  b, 64, 40, 40
        #       output3  b, 64, 20, 20
        #-------------------------------------------#
        fpn = self.fpn(out)

        #-------------------------------------------#
        #   ssh调整后宽高不变
        #-------------------------------------------#
        feature1 = self.ssh1(fpn[0])    # 1, 64, 80, 80
        feature2 = self.ssh2(fpn[1])    # 1, 64, 40, 40
        feature3 = self.ssh3(fpn[2])    # 1, 64, 20, 20
        features = [feature1, feature2, feature3]

        #-------------------------------------------#
        #   分别计算ClassHead,BboxHead,LandmarkHead,
        #   计算后的维度是 [b,特征框数量,2/4/10] 所以能在维度1上拼接
        #   将所有结果进行堆叠
        #-------------------------------------------#
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications  = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)      # [1, 67200, 2]
        ldm_regressions  = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.mode == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        # 框的调整参数,是否包含人脸,5个坐标点
        return output
