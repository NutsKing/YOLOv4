# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2020-12-21 11:21
# @Software: PyCharm
# @Brief:

from core.loss import YoloLoss
from nets.yolo import yolo_body
import config.config as cfg
from core.dataReader import DataReader

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers, callbacks


class CosineAnnealSchedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, epoch, train_step, lr_max, lr_min, warmth_rate=0.2):
        """
        学习率调节函数
        :param epoch: 训练轮次
        :param train_step: 一轮训练次数
        :param lr_max: 最大学习率
        :param lr_min: 最小学习率
        :param warmth_rate: 预热轮次的占比
        """
        super(CosineAnnealSchedule, self).__init__()

        self.total_step = epoch * train_step
        self.warm_step = self.total_step * warmth_rate
        self.lr_max = lr_max
        self.lr_min = lr_min

    @tf.function
    def __call__(self, step):
        if step < self.warm_step:
            lr = self.lr_max / self.warm_step * step
        else:
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1.0 + tf.cos((step - self.warm_step) / self.total_step * np.pi)
            )

        return lr


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr("float32")
    return lr


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 读取数据
    reader = DataReader(cfg.annotation_path, cfg.input_shape, cfg.batch_size, cfg.data_augmentation)
    train_data = reader.generate('train')
    validation_data = reader.generate('validation')
    train_steps = len(reader.train_lines) // cfg.batch_size
    validation_steps = len(reader.validation_lines) // cfg.batch_size

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(reader.train_lines),
                                                                               len(reader.validation_lines),
                                                                               cfg.batch_size))

    learning_rate = CosineAnnealSchedule(cfg.epochs, train_steps, cfg.lr, cfg.lr / 1e+4)
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    yolo_loss = [YoloLoss(cfg.anchors[mask]) for mask in cfg.anchor_masks]

    model = yolo_body()
    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer, loss=yolo_loss)

    train_by_fit(model, train_data, validation_data, train_steps, validation_steps)


def train_by_fit(model, train_datasets, valid_datasets, train_steps, valid_steps):
    """
    使用fit方式训练，更规范的添加callbacks参数
    :param model: 训练模型
    :param train_datasets: 训练集数据
    :param valid_datasets: 验证集数据
    :param train_steps: 迭代一个epoch的轮次
    :param valid_steps: 同上
    :return: None
    """
    cbk = [
        callbacks.ModelCheckpoint('./model/yolov4_{val_loss:.04f}.h5', save_best_only=True, save_weights_only=True)
    ]

    model.fit(train_datasets,
              steps_per_epoch=max(1, train_steps),
              validation_data=valid_datasets,
              validation_steps=max(1, valid_steps),
              epochs=cfg.epochs,
              callbacks=cbk)


if __name__ == '__main__':
    main()
