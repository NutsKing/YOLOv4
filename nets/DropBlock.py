# -*- coding: utf-8 -*-
# @File : DropBlock.py
# @Author: Runist
# @Time : 2020-12-24 16:44
# @Software: PyCharm
# @Brief: 自定义Layer实现DropBlock
from tensorflow.keras import layers, backend
import numpy as np
import copy
import time


class DropBlock2D(layers.Layer):

    def __init__(self, block_size, keep_prob, **kwargs):
        """
        DropBlock层
        :param block_size: dropblock的边长
        :param keep_prob: 保留的概率
        :param kwargs: 父类的参数
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.height, self.width = None, None

    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]

    def _get_gamma(self):
        """
        获取Bernoulli分布的概率
        :return: gamma值
        """

        x1 = (1.0 - self.keep_prob) / (self.block_size ** 2)
        x2 = self.height * self.width / ((self.height - self.block_size + 1.0) * (self.width - self.block_size + 1.0))
        gamma = x1 * x2

        return gamma

    def _get_seed_shape(self):
        """
        获取Bernoulli分布的生成矩阵
        :return: seed_shape
        """
        padding = self._get_padding()
        seed_shape = (self.height - (padding * 2), self.width - (padding * 2))

        return seed_shape

    def _get_padding(self):
        """
        获取drop中心点矩阵的填充值（补齐到与特征层相同shape的值）
        :return:
        """
        padding = self.block_size // 2

        return padding

    def call(self, inputs, training=None):
        """
        :param inputs: 上一特征层输出
        :param training: 是否训练
        :return: 训练阶段返回outputs, 预测阶段返回inputs
        """
        outputs = inputs

        # 计算gamma值
        gamma = self._get_gamma()
        shape = self._get_seed_shape()
        padding = self._get_padding()

        # 生成全0或1的矩阵（在这个二维矩阵中选择值为1作为drop的中心点）
        sample = np.random.binomial(n=1, size=shape, p=gamma)

        # 将矩阵0、1翻转，则值为0的中心作为drop中心
        sample = 1 - sample

        # 用1将矩阵填充至与该层一样的大小
        mask = np.pad(sample, pad_width=padding, mode='constant', constant_values=1)

        index = np.argwhere(mask == 0)
        # 对于每个0，创建形状为（block_size x block_size）的空间正方形蒙版
        for idx in index:
            i, j = idx
            mask[i-padding: i+padding+1, j-padding: j+padding+1] = 0

        mask = np.expand_dims(mask, axis=-1)

        # 先把mask扩充为与特征层同shape，然后进行乘法drop操作
        outputs = outputs * np.repeat(mask, inputs.shape[-1], -1)

        # 进行归一化操作，保证该层具有相同的均值和方差
        count = np.prod(mask.shape)     # 计算所有元素的乘积
        count_ones = np.count_nonzero(mask)    # 计算1的总数
        outputs = outputs * count / count_ones

        # backend.in_train_phase(x, alt, training=None)的作用是训练阶段返回x，在预测阶段返回alt
        return backend.in_train_phase(outputs, inputs, training=training)

    def get_config(self):
        config = {'block_size': self.block_size,
                  'gamma': self.gamma,
                  'seed': self.seed}
        base_config = super(DropBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

