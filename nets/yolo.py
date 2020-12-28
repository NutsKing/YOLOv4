# -*- coding: utf-8 -*-
# @File : yolo.py
# @Author: Runist
# @Time : 2020-12-21 15:38
# @Software: PyCharm
# @Brief:


from tensorflow.keras import layers, models
import tensorflow as tf
import config.config as cfg
from nets.csp_darknet import darknet_body, DarknetConv2D_BN_Leaky


def make_last_layers(inputs, num_filter):
    """
    5次（conv + bn + leaky激活）
    2次（conv + bn + leaky激活）
    :param inputs: 输入
    :param num_filter: 卷积核个数
    :return: x
    """
    x = DarknetConv2D_BN_Leaky(inputs, num_filter, kernel_size=1)
    x = DarknetConv2D_BN_Leaky(x, num_filter * 2, kernel_size=3)
    x = DarknetConv2D_BN_Leaky(x, num_filter, kernel_size=1)
    x = DarknetConv2D_BN_Leaky(x, num_filter * 2, kernel_size=3)
    output_5 = DarknetConv2D_BN_Leaky(x, num_filter, kernel_size=1)

    x = DarknetConv2D_BN_Leaky(output_5, num_filter * 2, kernel_size=3)
    output_7 = DarknetConv2D_BN_Leaky(x, len(cfg.anchor_masks[0]) * (cfg.num_classes+5), 1, bn=False)

    return output_5, output_7


def SPP_net(inputs):
    """
    SPP结构，使得图片可以是任意大小，但输出是一样的。
    对图片进行三次MaxPooling2D，得到不同的感受野，在进行堆叠。得到原来的通道数x4的输出层
    :param inputs:
    :return:
    """
    # 使用了SPP结构，即不同尺度的最大池化后堆叠。
    maxpool1 = layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(inputs)
    maxpool2 = layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(inputs)
    maxpool3 = layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(inputs)
    output = layers.Concatenate()([maxpool1, maxpool2, maxpool3, inputs])

    return output


def Conv2D_Upsample(inputs, num_filter):
    """
    1次（conv + bn + leaky激活） + 上采样
    :param inputs: 输入层
    :param num_filter: 卷积核个数
    :return: x
    """
    x = DarknetConv2D_BN_Leaky(inputs, num_filter, kernel_size=1)
    x = layers.UpSampling2D(2)(x)

    return x


def yolo_feat_reshape(feat):
    """
    处理一下y_pred的数据，reshape，从b, 13, 13, 75 -> b, 13, 13, 3, 25
    在Keras.model编译前处理是为了loss计算上能匹配
    :param feat:
    :return:
    """
    grid_size = tf.shape(feat)[1]
    reshape_feat = tf.reshape(feat, [-1, grid_size, grid_size, len(cfg.anchor_masks[0]), cfg.num_classes + 5])

    return reshape_feat


def yolo_head(y_pred, anchors, calc_loss=False):
    """
    处理一下y_pred的数据，reshape，从b, 13, 13, 75 -> b, 13, 13, 3, 25
    另外，取名为head是有意义的。因为目标检测大多数分为 - Backbone - Detection head两个部分
    :param y_pred: 预测数据
    :param anchors: 其中一种大小的先验框（总共三种）
    :param calc_loss: 是否计算loss，该函数可以在直接预测的地方用
    :return:
        bbox: 存储了x1, y1 x2, y2的坐标 shape(b, 13, 13 ,3, 4)
        objectness: 该分类的置信度 shape(b, 13, 13 ,3, 1)
        class_probs: 存储了20个分类在sigmoid函数激活后的数值 shape(b, 13, 13 ,3, 20)
        pred_xywh: 把xy(中心点),wh shape(b, 13, 13 ,3, 4)
    """
    grid_size = tf.shape(y_pred)[1]

    # tf.spilt的参数对应：2-(x,y) 2-(w,h) 1-置信度 classes=20-分类数目的得分
    box_xy, box_wh, confidence, class_probs = tf.split(y_pred, (2, 2, 1, cfg.num_classes), axis=-1)
    # 举例：box_xy (13, 13, 3, 2) 3是指三个框，2是xy，其他三个输出类似

    # sigmoid是为了让tx, ty在[0, 1]，防止偏移过多，使得中心点落在一个网络单元格中，这也是激活函数的作用（修正）
    # 而对confidence和class_probs使用sigmoid是为了得到0-1之间的概率
    box_xy = tf.sigmoid(box_xy)
    confidence = tf.sigmoid(confidence)
    class_probs = tf.sigmoid(class_probs)

    # !!! grid[x][y] == (y, x)
    # sigmoid(x) + cx，在这里看，生成grid的原因是要和y_true的格式对齐。
    # 而且加上特征图就是13x13 26x26...一个特征图上的点，就预测一个结果。
    grid_y = tf.tile(tf.reshape(tf.range(grid_size), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_size), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)  # [gx, gy, 1, 2]
    grid = tf.cast(grid, tf.float32)

    # 把xy, wh归一化成比例
    # box_xy(b, 13, 13, 3, 2)  grid(13, 13, 1, 2)  grid_size shape-()-13
    # box_wh(b, 13, 13, 3, 2)  anchors_tensor(1, 1, 1, 3, 2)
    box_xy = (box_xy + grid) / tf.cast(grid_size, tf.float32)
    # 要注意，xy除去的是13，wh除去的416，是因为下面wh用的也是416(如果xywh不归一化，和概率值一起训练肯定不收敛啊)
    box_wh = tf.exp(box_wh) * anchors / cfg.input_shape
    # 最后 box_xy、box_wh 都是 (b, 13, 13, 3, 2)

    # 把xy,wh 合并成pred_box在最后一个维度上（axis=-1）
    pred_xywh = tf.concat([box_xy, box_wh], axis=-1)  # original xywh for loss

    if calc_loss:
        return pred_xywh, grid

    return box_xy, box_wh, confidence, class_probs


def yolo_body():
    """
    yolov4的骨干网络部分，这里注释写的是输入情况是416x416的前提，如果输入大小时608x608的情况
    13x13 = 19x19、 26x26 = 38x38、 52x52 = 76x76
    对比yolov3（只有特征金字塔结构），在特征提取部分使用了PANet（下采样融合）的网络结构，加强了特征融合，提取更有效地特征
    :return: model
    """
    height, width = cfg.input_shape
    input_image = layers.Input(shape=(height, width, 3), dtype='float32')  # [b, 416, 416, 3]

    feat_52x52, feat_26x26, feat_13x13 = darknet_body(input_image)

    # 13x13 head
    # 三次卷积 + SPP + 三次卷积
    y13 = DarknetConv2D_BN_Leaky(feat_13x13, 512, kernel_size=1)
    y13 = DarknetConv2D_BN_Leaky(y13, 1024, kernel_size=3)
    y13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=1)
    y13 = SPP_net(y13)
    y13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=1)
    y13 = DarknetConv2D_BN_Leaky(y13, 1024, kernel_size=3)
    y13 = DarknetConv2D_BN_Leaky(y13, 512, kernel_size=1)
    y13_upsample = Conv2D_Upsample(y13, 256)

    # PANet
    # 26x26 head
    y26 = DarknetConv2D_BN_Leaky(feat_26x26, 256, kernel_size=1)
    y26 = layers.Concatenate()([y26, y13_upsample])
    y26, _ = make_last_layers(y26, 256)     # TODO 到时候裁剪模型时就要把这里改一下
    y26_upsample = Conv2D_Upsample(y26, 128)

    # 52x52 head and output
    y52 = DarknetConv2D_BN_Leaky(feat_52x52, 128, (1, 1))
    y52 = layers.Concatenate()([y52, y26_upsample])
    y52, output_52x52 = make_last_layers(y52, 128)

    # 26x26 output
    y52_downsample = layers.ZeroPadding2D(((1, 0), (1, 0)))(y52)
    y52_downsample = DarknetConv2D_BN_Leaky(y52_downsample, 256, kernel_size=3, strides=(2, 2))
    y26 = layers.Concatenate()([y52_downsample, y26])
    y26, output_26x26 = make_last_layers(y26, 256)

    # 13x13 output
    y26_downsample = layers.ZeroPadding2D(((1, 0), (1, 0)))(y26)
    y26_downsample = DarknetConv2D_BN_Leaky(y26_downsample, 512, kernel_size=3, strides=(2, 2))
    y13 = layers.Concatenate()([y26_downsample, y13])
    y13, output_13x13 = make_last_layers(y13, 512)

    # 这里output1、output2、output3的shape分别是52x52, 26x26, 13x13
    # 然后reshape为 从(b, size, size, 75) -> (b, size, size, 3, 25)
    output_52x52 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='feat52')(output_52x52)
    output_26x26 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='feat26')(output_26x26)
    output_13x13 = layers.Lambda(lambda x: yolo_feat_reshape(x), name='feat13')(output_13x13)

    return models.Model(input_image, [output_13x13, output_26x26, output_52x52])
