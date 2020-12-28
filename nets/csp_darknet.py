# -*- coding: utf-8 -*-
# @File : csp_darknet.py
# @Author: Runist
# @Time : 2020-12-21 15:27
# @Software: PyCharm
# @Brief: 骨干网络

from tensorflow.keras import layers, regularizers
from nets.DropBlock import DropBlock2D
import tensorflow as tf
import config.config as cfg


class Mish(layers.Layer):
    """
    Mish激活函数
    公式：
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: 任意的. 使用参数 `input_shape`
        - Output: 和输入一样的shape
    Examples:
        >> X_input = layers.Input(input_shape)
        >> X = Mish()(X_input)
    """
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    @staticmethod
    def call(inputs, **kwargs):
        return inputs * tf.tanh(tf.nn.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    @staticmethod
    def compute_output_shape(input_shape, **kwargs):
        return input_shape


def DarknetConv2D_BN_Mish(inputs, num_filter, kernel_size, strides=(1, 1), bn=True):
    """
    卷积 + 批归一化 + leaky激活，因为大量用到这样的结构，所以这样写
    :param inputs: 输入
    :param num_filter: 卷积个数
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :param bn: 是否使用批归一化
    :return: x
    """
    if strides == (1, 1) or strides == 1:
        padding = 'same'
    else:
        padding = 'valid'

    x = layers.Conv2D(num_filter, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      use_bias=not bn, kernel_regularizer=regularizers.l2(5e-4),  # 只有添加正则化参数，才能调用model.losses方法
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01))(inputs)

    if bn:
        x = layers.BatchNormalization()(x)
        x = Mish()(x)

    return x


def DarknetConv2D_BN_Leaky(inputs, num_filter, kernel_size, strides=(1, 1), bn=True):
    """
    卷积 + 批归一化 + leaky激活，因为大量用到这样的结构，
    其中名字的管理比较麻烦，所以添加了函数内部变量
    :param inputs: 输入
    :param num_filter: 卷积个数
    :param kernel_size: 卷积核大小
    :param strides: 步长
    :param bn: 是否使用批归一化
    :return: x
    """

    if strides == (1, 1) or strides == 1:
        padding = 'same'
    else:
        padding = 'valid'

    x = layers.Conv2D(num_filter, kernel_size=kernel_size,
                      strides=strides, padding=padding,              # 这里的参数是只l2求和之后所乘上的系数
                      use_bias=not bn, kernel_regularizer=regularizers.l2(5e-4),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01))(inputs)

    if bn:
        x = layers.BatchNormalization()(x)
        # alpha是x < 0时，变量系数
        x = layers.LeakyReLU(alpha=0.1)(x)

    return x


def resblock_body(inputs, filters, num_blocks, all_narrow=True):
    """
    残差块
    ZeroPadding + conv + nums_filters 次 darknet_block
    :param inputs: 上一层输出
    :param filters: conv的卷积核个数，每次残差块是不一样的
    :param num_blocks: 有几个这样的残差块
    :param all_narrow:
    :return: 卷积结果
    """
    # 进行长和宽的压缩(减半，这一部分和原本的Darknet53一样)
    preconv1 = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    preconv1 = DarknetConv2D_BN_Mish(preconv1, filters, kernel_size=3, strides=(2, 2))

    # 生成一个大的残差边(对应左边的shortcut)
    shortconv = DarknetConv2D_BN_Mish(preconv1, filters//2 if all_narrow else filters, kernel_size=1)

    # 主干部分的卷积(对应右边的卷积)
    mainconv = DarknetConv2D_BN_Mish(preconv1, filters//2 if all_narrow else filters, kernel_size=1)
    # 1x1卷积对通道数进行整合->3x3卷积提取特征，使用残差结构
    for i in range(num_blocks):
        x = DarknetConv2D_BN_Mish(mainconv, filters//2, kernel_size=1)
        x = DarknetConv2D_BN_Mish(x, filters//2 if all_narrow else filters, kernel_size=3)

        mainconv = layers.Add()([mainconv, x])

    # 1x1卷积后和残差边堆叠
    postconv = DarknetConv2D_BN_Mish(mainconv, filters//2 if all_narrow else filters, kernel_size=1)
    route = layers.Concatenate()([postconv, shortconv])

    # 最后对通道数进行整合
    output = DarknetConv2D_BN_Mish(route, filters, (1, 1))
    return output


def darknet_body(inputs):
    """
    darknet53是yolov4的特征提取网络，输出三个大小的特征层
    :param inputs: 输入图片[n, 416, 416, 3]
    :return:
    """
    x = DarknetConv2D_BN_Mish(inputs, 32, 3)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = DropBlock2D(block_size=7, keep_prob=0.9)(x)
    feat_52x52 = x

    x = resblock_body(x, 512, 8)
    x = DropBlock2D(block_size=7, keep_prob=0.9)(x)
    feat_26x26 = x

    x = resblock_body(x, 1024, 4)
    x = DropBlock2D(block_size=7, keep_prob=0.9)(x)
    feat_13x13 = x

    return feat_52x52, feat_26x26, feat_13x13

