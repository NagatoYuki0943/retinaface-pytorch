import numpy as np
import torch
from torchvision.ops import nms


#-----------------------------------------------------------------#
#   将输出调整为相对于原图的大小
#-----------------------------------------------------------------#
def retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape   = image_shape*np.min(input_shape/image_shape)

    offset      = (input_shape - new_shape) / 2. / input_shape
    scale       = input_shape / new_shape

    scale_for_boxs      = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0]]

    offset_for_boxs         = [offset[1], offset[0], offset[1],offset[0]]
    offset_for_landmarks    = [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0]]

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result

#-----------------------------#
#   中心解码，宽高解码
#   对先验框调整获得预测框
#-----------------------------#
def decode(loc, priors, variances):
    """
    loc:    预测值 [b, 4]
    priors: 原始值 [b, 4]
    variances: 标准化预测值 [2]
    return: [b, 4] 4指的是x1,y1,x2,y2
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],   # 中心调整: 原始坐标 + 预测值 * variance * 原始宽高
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)    # 宽高调整: 原始宽高 * e^(预测值*variance)
    # 转换为左上角右下角形式返回
    boxes[:, :2] -= boxes[:, 2:] / 2    # 左上角
    boxes[:, 2:] += boxes[:, :2]        # 右下角
    return boxes

#-----------------------------#
#   关键点解码
#   对先验框中心进行调整,获得5个关键点
#   类似中心调整: 原始坐标 + 预测值 * variance * 原始宽高
#-----------------------------#
def decode_landm(pre, priors, variances):
    """
    pre:    预测值  [b,10]
    priors: 原始值  [b, 4]
    variances: 标准化预测值 [2]
    return: 5个坐标点 [b,10]
    """
    # priors[:, :2]代表中心坐标,全在它的基础上进行调整
    landms = torch.cat((priors[:, :2] + pre[:,  :2] * variances[0] * priors[:, 2:],  # 类似中心调整: 原始坐标 + 预测值 * variance * 原始宽高
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10]* variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms

#-----------------------------#
#   交并比
#-----------------------------#
def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)

    iou = inter_area/np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou

#-----------------------------#
#   非极大抑制
#-----------------------------#
def non_max_suppression(detection, conf_thres=0.5, nms_thres=0.3):
    """
    detection:
                [0,1,2,3] 坐标
                [4] 置信度
                [5-14] 5个关键点坐标
    """
    #------------------------------------------#
    #   1、找出该图片中得分大于门限函数的框。
    #   在进行重合框筛选前就
    #   进行得分的筛选可以大幅度减少框的数量。
    #------------------------------------------#
    mask        = detection[:, 4] >= conf_thres # 置信度
    detection   = detection[mask]               # 保留一部分

    if len(detection) <= 0:
        return []

    #------------------------------------------#
    #   使用官方自带的非极大抑制会速度更快一些！
    #------------------------------------------#
    keep = nms(
        detection[:, :4],   # 坐标
        detection[:, 4],    # 先验框置信度 * 种类置信度 结果是1维数据
        nms_thres
    )
    best_box = detection[keep]

    # best_box = []
    # scores = detection[:, 4]
    # # 2、根据得分对框进行从大到小排序。
    # arg_sort = np.argsort(scores)[::-1]
    # detection = detection[arg_sort]

    # while np.shape(detection)[0]>0:
    #     # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
    #     best_box.append(detection[0])
    #     if len(detection) == 1:
    #         break
    #     ious = iou(best_box[-1], detection[1:])
    #     detection = detection[1:][ious<nms_thres]
    return best_box.cpu().numpy()
