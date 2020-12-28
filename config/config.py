# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020-12-21 14:23
# @Software: PyCharm
# @Brief: 配置文件
import numpy as np

# 相关路径信息
annotation_path = "./config/train.txt"

# 获得分类名
class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 模型相关参数
num_classes = len(class_names)
input_shape = (416, 416)
lr = 1e-4
batch_size = 8
epochs = 100

# nms与最低分数阈值
ignore_thresh = 0.5
iou_threshold = 0.3

# 标签处理
label_smooth = 0.05

# 数据处理
valid_rate = 0.1
data_augmentation = "all"  # mosaic，random(单张图片的数据增强)，mixup将两张图片进行加权混合，normal(不增强，只进行简单填充)

# 先验框个数、先验框信息 和 对应索引
anchors = np.array([(5, 9), (9, 16), (15, 26),
                    (23, 38), (34, 53), (49, 80),
                    (80, 119), (130, 179), (200, 243)],
                   np.float32)

anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_bbox = len(anchor_masks[0])
