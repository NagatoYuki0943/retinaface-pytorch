"""
先验框
"""

from itertools import product as product
from math import ceil

import torch


class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super().__init__()
        #------------------------------------------------------------------#
        #   先验框大小,由浅到深,浅的特征层框比较小,深的特征层框比较大
        #------------------------------------------------------------------#
        self.min_sizes  = cfg['min_sizes']
        #------------------------------------------------------------------#
        #   三个有效特征层对于长宽压缩的倍数
        #------------------------------------------------------------------#
        self.steps      = cfg['steps']
        #------------------------------------------------------------------#
        #   是否生成先验框后clip到0~1之间
        #------------------------------------------------------------------#
        self.clip       = cfg['clip']
        #---------------------------#
        #   输入图片的尺寸
        #---------------------------#
        self.image_size = image_size
        #---------------------------#
        #   三个有效特征层高和宽
        #---------------------------#
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    """获得先验框"""
    def get_anchors(self):
        anchors = []
        # 循环三个特征层
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            #-----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            #-----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                # 先验框映射到网格点上
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    # 添加到空列表中
                    for cy, cx in product(dense_cy, dense_cx):
                        # 中心宽高
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
