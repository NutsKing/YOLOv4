# -*- coding: utf-8 -*-
# @File : convert.py
# @Author: Runist
# @Time : 2020/5/6 9:38
# @Software: PyCharm
# @Brief: 将Darknet框架下训练的模型转换成Keras中h5格式


import io
import os
import numpy as np
import configparser
from collections import defaultdict

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from keras import backend as K
from tensorflow.keras.layers import (Conv2D, Input, ZeroPadding2D, Add, UpSampling2D, MaxPooling2D, Concatenate)
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import config.config as cfg
from nets.model import Mish


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compability with configparser.
    """
    # 当Key不存在时，会引发‘KeyError’异常。为了避免这种情况的发生，可以使用collections类中的defaultdict()方法来为字典提供默认值
    section_counters = defaultdict(int)
    output_stream = io.StringIO()

    with open(config_file) as file:
        for line in file:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def main():
    config_path = "yolov4.cfg"
    output_path = cfg.pretrain_weights_path
    weights_path = 'yolov4.weights'
    weights_only = False

    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(shape=(3,), dtype='int32', buffer=weights_file.read(12))

    # 读取文件类型
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        # open.read方法可以从文件中读取数据，参数表示要从文件中读取的数据集长度（字符个数）
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    # configparser是用来读取cfg文件的包
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating Keras model.')
    h, w = cfg.input_shape
    input_layer = Input(shape=(h, w, 3))
    prev_layer = input_layer
    all_layers = []

    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    out_index = []
    i = 0

    for section in cfg_parser.sections():
        print(i)
        i += 1
        print('Parsing section {}'.format(section))
        # 如果是以convolutional开头的，下面同理
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # 设置权重
            # Darknet  卷积权重顺序是:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            # 通过读取到cfg值设置weight shape
            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape, "\n")

            conv_bias = np.ndarray(shape=(filters, ), dtype='float32', buffer=weights_file.read(filters * 4))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(shape=(3, filters), dtype='float32', buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,      # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]   # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))

            count += weights_size

            # DarkNet conv_weights是Caffe和Pytorch的顺序
            # (out_dim, in_dim, height, width)
            # 把他改成Tensorflow的样式:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [conv_weights, conv_bias]

            # Create Conv2D layer
            if stride > 1:
                # Darknet只填充左上，而不是采用Tensorflow的'same'的方式
                prev_layer = ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)

            conv_layer = (Conv2D(
                filters, (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=None,
                padding=padding))(prev_layer)

            if batch_normalize:
                conv_layer = (BatchNormalization(weights=bn_weight_list))(conv_layer)

            prev_layer = conv_layer

            # shortcut的activation是linear，也就是没有激活函数
            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'mish':
                act_layer = Mish()(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('yolo'):
            out_index.append(len(all_layers) - 1)
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass

        else:
            raise ValueError('Unsupported section header type: {}'.format(section))

    # Create and save model.
    if len(out_index) == 0:
        out_index.append(len(all_layers) - 1)

    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    model.summary()

    if weights_only:
        model.save_weights(output_path)
        print('Saved Keras weights to {}'.format(output_path))
    else:
        model.save(output_path)
        print('Saved Keras model to {}'.format(output_path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count + remaining_weights))

    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))


if __name__ == '__main__':
    main()