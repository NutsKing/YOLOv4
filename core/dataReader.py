# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020-12-21 11:23
# @Software: PyCharm
# @Brief: 数据读取


import tensorflow as tf
import numpy as np
import config.config as cfg
from PIL import Image
import cv2 as cv


class DataReader:

    def __init__(self, data_path, input_shape, batch_size, data_aug, max_boxes=100):
        self.data_path = data_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.max_boxes = max_boxes
        self.data_aug = data_aug
        self.train_lines, self.validation_lines = self.read_data_and_split_data()

    def read_data_and_split_data(self):
        """
        读取数据，并按照训练集和验证集分割数据
        :return: 训练数据，验证数据
        """
        with open(self.data_path, "r", encoding='utf-8') as f:
            files = f.readlines()

        split = int(cfg.valid_rate * len(files))
        train = files[split:]
        valid = files[:split]

        return train, valid

    def get_data(self, annotation_line):
        """
        获取数据（不增强）
        :param annotation_line: 一行数据（图片路径 + 坐标）
        :return: image，box_data
        """
        line = annotation_line.split()
        image = Image.open(line[0])
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image_width, image_height = image.size
        input_width, input_height = self.input_shape
        scale = min(input_width / image_width, input_height / image_height)

        new_width = int(image_width * scale)
        new_height = int(image_height * scale)

        image = image.resize((new_width, new_height), Image.BICUBIC)
        new_image = Image.new('RGB', self.input_shape, (128, 128, 128))
        new_image.paste(image, ((input_width - new_width)//2, (input_height - new_height)//2))

        image = np.asarray(new_image) / 255

        dx = (input_width - new_width) / 2
        dy = (input_height - new_height) / 2

        # 为填充过后的图片，矫正box坐标，如果没有box需要检测annotation文件
        if len(box) <= 0:
            raise Exception("{} doesn't have any bounding boxes.".format(image_path))

        box_data = np.zeros((self.max_boxes, 5), dtype='float32')
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy

        if len(box) > self.max_boxes:
            box = box[:self.max_boxes]

        box_data[:len(box)] = box

        return image, box_data

    def get_random_data(self, annotation_line, hue=.1, sat=1.5, val=1.5):
        """
        数据增强（改变长宽比例、大小、亮度、对比度、颜色饱和度）
        :param annotation_line: 一行数据
        :param hue: 色调抖动
        :param sat: 饱和度抖动
        :param val: 明度抖动
        :return: image, box_data
        """
        line = annotation_line.split()
        image_path = line[0]
        image = Image.open(image_path)

        image_width, image_height = image.size
        input_width, input_height = self.input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 随机生成缩放比例，缩小或者放大
        scale = rand(0.8, 1.5)
        # 随机变换长宽比例
        new_ar = input_width / input_height * rand(0.7, 1.3)

        if new_ar < 1:
            new_height = int(scale * input_height)
            new_width = int(new_height * new_ar)
        else:
            new_width = int(scale * input_width)
            new_height = int(new_width / new_ar)

        image = image.resize((new_width, new_height), Image.BICUBIC)

        dx = rand(0, (input_width - new_width))
        dy = rand(0, (input_height - new_height))
        new_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
        new_image.paste(image, (int(dx), int(dy)))
        image = new_image

        # 翻转图片
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 图像增强
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv.cvtColor(np.array(image, np.float32)/255, cv.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image = cv.cvtColor(x, cv.COLOR_HSV2RGB)

        box_data = np.zeros((self.max_boxes, 5))
        # 为填充过后的图片，矫正box坐标，如果没有box需要检测annotation文件
        if len(box) <= 0:
            raise Exception("{} doesn't have any bounding boxes.".format(image_path))

        box[:, [0, 2]] = box[:, [0, 2]] * new_width / image_width + dx
        box[:, [1, 3]] = box[:, [1, 3]] * new_height / image_height + dy
        # 若翻转了图像，框也需要翻转
        if flip:
            box[:, [0, 2]] = input_width - box[:, [2, 0]]

        # 定义边界
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > input_width] = input_width
        box[:, 3][box[:, 3] > input_height] = input_height

        # 计算新的长宽
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]

        # 去除无效数据
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        if len(box) > self.max_boxes:
            box = box[:self.max_boxes]

        box_data[:len(box)] = box

        return image, box_data

    def get_mixup_data(self, annotation_line, hue=.1, sat=1.5, val=1.5):
        """
        使用mixup进行数据增强
        :param annotation_line: 2行数据
        :param hue: 色域变换的h色调
        :param sat: 饱和度S
        :param val: 明度V
        :return: image, box_data
        """
        input_height, input_width = self.input_shape
        image_data = []
        box_data = []

        for i in range(2):
            line = annotation_line[i].split()
            image_path = line[0]
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

            image = Image.open(image_path)
            image = image.convert("RGB")
            # 图片的大小
            image_width, image_height = image.size

            # 是否翻转图片
            flip = rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # 只对图像进行正常的resize
            scale = min(input_width / image_width, input_height / image_height)
            new_width = int(image_width * scale)
            new_height = int(image_height * scale)
            image = image.resize((new_width, new_height), Image.BICUBIC)
            new_image = Image.new('RGB', self.input_shape, (128, 128, 128))
            new_image.paste(image, ((input_width - new_width)//2, (input_height - new_height)//2))
            image = new_image

            dx = (input_width - new_width) / 2
            dy = (input_height - new_height) / 2

            # 图像增强
            hue = rand(-hue, hue)
            sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
            val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
            x = cv.cvtColor(np.array(image, np.float32)/255, cv.COLOR_RGB2HSV)
            x[..., 0] += hue*360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image = cv.cvtColor(x, cv.COLOR_HSV2RGB)

            # 为填充过后的图片，矫正box坐标，如果没有box需要检测annotation文件
            if len(box) <= 0:
                raise Exception("{} doesn't have any bounding boxes.".format(image_path))

            # box_data = np.zeros((self.max_boxes, 5), dtype='float32')
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy

            image_data.append(image)
            box_data.append(box)

        # alpha = rand(0.3, 0.7)
        alpha = 0.5
        image = alpha * image_data[0] + (1 - alpha) * image_data[1]
        new_boxes = np.concatenate((box_data[0], box_data[1]), axis=0)

        # 将box进行调整
        box_data = np.zeros((self.max_boxes, 5))

        if len(new_boxes) > self.max_boxes:
            new_boxes = new_boxes[:self.max_boxes]
        box_data[:len(new_boxes)] = new_boxes

        return image, box_data

    def get_mosaic_data(self, annotation_line, hue=.1, sat=1.5, val=1.5):
        """
        mosaic数据增强方式
        :param annotation_line: 4行图像信息数据
        :param hue: 色域变换的h色调
        :param sat: 饱和度S
        :param val: 明度V
        :return: image, box_data
        """
        input_height, input_width = self.input_shape

        min_offset_x = 0.45
        min_offset_y = 0.45
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_data = []
        box_data = []

        # 定义分界线，用列表存储
        place_x = [0, 0, int(input_width * min_offset_x), int(input_width * min_offset_x)]
        place_y = [0, int(input_height * min_offset_y), int(input_width * min_offset_y), 0]
        for i in range(4):
            line = annotation_line[i].split()
            image_path = line[0]
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

            # 打开图片
            image = Image.open(image_path)
            image = image.convert("RGB")
            # 图片的大小
            image_width, image_height = image.size

            # 是否翻转图片
            flip = rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = image_width - box[:, [2, 0]]

            # 对输入进来的图片进行缩放
            scale = rand(scale_low, scale_high)
            new_height = int(scale * image_height)
            new_width = int(scale * image_width)
            image = image.resize((new_width, new_height), Image.BICUBIC)

            # 图像增强
            hue = rand(-hue, hue)
            sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
            val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
            x = cv.cvtColor(np.array(image, np.float32)/255, cv.COLOR_RGB2HSV)
            x[..., 0] += hue*360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image = cv.cvtColor(x, cv.COLOR_HSV2RGB)

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[i]
            dy = place_y[i]

            mosaic_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
            mosaic_image.paste(image, (dx, dy))
            mosaic_image = np.array(mosaic_image) / 255

            # 为填充过后的图片，矫正box坐标，如果没有box需要检测annotation文件
            if len(box) <= 0:
                raise Exception("{} doesn't have any bounding boxes.".format(image_path))

            np.random.shuffle(box)
            # 重新计算box的宽高 乘上尺度 加上偏移
            box[:, [0, 2]] = box[:, [0, 2]] * new_width / image_width + dx
            box[:, [1, 3]] = box[:, [1, 3]] * new_height / image_height + dy

            # 定义边界(box[:, 0:2] < 0的到的是Bool型的列表，True值置为边界)
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > input_width] = input_width
            box[:, 3][box[:, 3] > input_height] = input_height

            # 计算新的长宽
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]

            # 去除无效数据
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box = np.array(box, dtype=np.float)

            image_data.append(mosaic_image)
            box_data.append(box)

        # 随机选取分界线，将图片放上去
        cutx = np.random.randint(int(input_width * min_offset_x), int(input_width * (1 - min_offset_x)))
        cuty = np.random.randint(int(input_height * min_offset_y), int(input_height * (1 - min_offset_y)))

        mosaic_image = np.zeros([input_height, input_width, 3])
        mosaic_image[:cuty, :cutx] = image_data[0][:cuty, :cutx]
        mosaic_image[cuty:, :cutx] = image_data[1][cuty:, :cutx]
        mosaic_image[cuty:, cutx:] = image_data[2][cuty:, cutx:]
        mosaic_image[:cuty, cutx:] = image_data[3][:cuty, cutx:]

        # 对框进行坐标系的处理
        new_boxes = self.merge_boxes(box_data, cutx, cuty)

        # 将box进行调整
        box_data = np.zeros((self.max_boxes, 5))
        if len(new_boxes) > 0:
            if len(new_boxes) > self.max_boxes:
                new_boxes = new_boxes[:self.max_boxes]
            box_data[:len(new_boxes)] = new_boxes

        return mosaic_image, box_data

    @staticmethod
    def merge_boxes(boxes, cutx, cuty):
        """
        四张图的box的合并，合并前是都是基于0坐标的Box。现在要将box合并到同一个坐标系下
        :param boxes: 真实框
        :param cutx: 分界线x坐标
        :param cuty: 分解线y坐标
        :return: merge_box合并后的坐标
        """
        merge_box = []
        for i in range(len(boxes)):
            for box in boxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    # 如果左上角的坐标比分界线大，就不要了
                    if y1 > cuty or x1 > cutx:
                        continue
                    # 分界线在y1和y2之间。就取cuty
                    if y2 >= cuty >= y1:
                        y2 = cuty
                        # 类似于这样的宽或高太短的就不要了
                        if y2 - y1 < 5:
                            continue
                    if x2 >= cutx >= x1:
                        x2 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue

                    if y2 >= cuty >= y1:
                        y1 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx >= x1:
                        x2 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue

                    if y2 >= cuty >= y1:
                        y1 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx >= x1:
                        x1 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue

                    if y2 >= cuty >= y1:
                        y2 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx >= x1:
                        x1 = cutx
                        if x2 - x1 < 5:
                            continue

                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_box.append(tmp_box)

        return merge_box

    def process_true_bbox(self, box_data):
        """
        对真实框处理，首先会建立一个13x13，26x26，52x52的特征层，具体的shape是
        [b, n, n, 3, 25]的特征层，也就意味着，一个特征层最多可以存放n^2个数据
        :param box_data: 实际框的数据
        :return: 处理好后的 y_true
        """

        true_boxes = np.array(box_data, dtype='float32')
        input_shape = np.array(self.input_shape, dtype='int32')  # 416,416

        # “...”(ellipsis)操作符，表示其他维度不变，只操作最前或最后1维。读出xy轴，读出长宽
        # true_boxes[..., 0:2] 是左上角的点 true_boxes[..., 2:4] 是右上角的点
        # 计算中心点 和 宽高
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        # 实际的宽高 / 416 转成比例
        true_boxes[..., 0:2] = boxes_xy / input_shape
        true_boxes[..., 2:4] = boxes_wh / input_shape

        # 生成3种特征大小的网格
        grid_shapes = [input_shape // [32, 16, 8][i] for i in range(cfg.num_bbox)]
        # 创建3个特征大小的全零矩阵，[(b, 13, 13, 3, 25), ... , ...]存在列表中
        y_true = [np.zeros((self.batch_size,
                            grid_shapes[i][0], grid_shapes[i][1], cfg.num_bbox, 5 + cfg.num_classes),
                           dtype='float32') for i in range(cfg.num_bbox)]

        # 计算哪个先验框比较符合 真实框的Gw,Gh 以最高的iou作为衡量标准
        # 因为先验框数据没有坐标，只有宽高，那么现在假设所有的框的中心在（0，0），宽高除2即可。（真实框也要做一样的处理才能匹配）
        anchors = np.expand_dims(cfg.anchors, 0)
        anchor_rightdown = anchors / 2.     # 网格中心为原点(即网格中心坐标为(0,0)),　计算出anchor 右下角坐标
        anchor_leftup = -anchor_rightdown     # 计算anchor 左上角坐标

        # 长宽要大于0才有效,也就是那些为了补齐到max_boxes大小的0数据无效
        # 返回一个列表，大于0的为True，小于等于0的为false
        # 选择具体一张图片，valid_mask存储的是true or false，然后只选择为true的行
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(self.batch_size):
            wh = boxes_wh[b, valid_mask[b]]
            wh = np.expand_dims(wh, 1)  # 在第二维度插入1个维度，便于后面进行broadcast
            box_rightdown = wh / 2.
            box_leftup = -box_rightdown

            # 获取坐标
            intersect_leftup = np.maximum(box_leftup, anchor_leftup)
            intersect_rightdown = np.minimum(box_rightdown, anchor_rightdown)
            intersect_wh = np.maximum(intersect_rightdown - intersect_leftup, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            # 计算真实框与anchor的iou
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = cfg.anchors[..., 0] * cfg.anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # 设定一个iou值，只要真实框与先验框的iou大于这个值就可以当作正样本输入进去。
            # 因为负样本是不参与loss计算的，这就使得正负样本不均衡。放宽正样本的筛选条件，以提高正负样本的比例
            iou_masks = iou > 0.3
            written = [False] * len(iou)

            # 对每个框进行遍历
            for key, iou_mask in enumerate(iou_masks):
                # 获取符合iou阈值条件的索引
                true_iou_mask = np.where(iou_mask)[0]
                for value in true_iou_mask:

                    n = (cfg.num_bbox - 1) - value // len(cfg.anchor_masks)

                    # 保证value（先验框的索引）的在anchor_masks[n]中 且 iou 大于阈值
                    x = np.floor(true_boxes[b, key, 0] * grid_shapes[n][1]).astype('int32')
                    y = np.floor(true_boxes[b, key, 1] * grid_shapes[n][0]).astype('int32')

                    # 获取 先验框（二维列表）内索引，k就是对应的最好anchor
                    k = cfg.anchor_masks[n].index(value)
                    c = true_boxes[b, key, 4].astype('int32')

                    # 三个大小的特征层，逐一赋值
                    y_true[n][b, y, x, k, 0:4] = true_boxes[b, key, 0:4]
                    y_true[n][b, y, x, k, 4] = 1       # 置信度是1 因为含有目标
                    y_true[n][b, y, x, k, 5 + c] = 1   # 类别的one-hot编码，其他都为0

                    # 如果这个bbox已经写入真实框数据，那么就不必再在后续的best_anchor写入数据
                    written[key] = True

            # 如果前面根据iou筛选框，并没有合适的框，则这一步计算出最匹配iou的作为先验框
            best_anchors = np.argmax(iou, axis=-1)
            for key, value in enumerate(best_anchors):
                n = (cfg.num_bbox - 1) - value // len(cfg.anchor_masks)
                # 如果没有写入，就写入最匹配的anchor
                if not written[key]:
                    x = np.floor(true_boxes[b, key, 0] * grid_shapes[n][1]).astype('int32')
                    y = np.floor(true_boxes[b, key, 1] * grid_shapes[n][0]).astype('int32')

                    # 获取 先验框（二维列表）内索引，k就是对应的最好anchor
                    k = cfg.anchor_masks[n].index(value)
                    c = true_boxes[b, key, 4].astype('int32')

                    # 三个大小的特征层，逐一赋值
                    y_true[n][b, y, x, k, 0:4] = true_boxes[b, key, 0:4]
                    y_true[n][b, y, x, k, 4] = 1       # 置信度是1 因为含有目标
                    y_true[n][b, y, x, k, 5 + c] = 1   # 类别的one-hot编码，其他都为0

        return y_true

    def generate(self, mode):
        """
        数据生成器
        :param mode: train or validation
        :return: image, rpn训练标签， 真实框数据
        """
        i = 0
        if mode == 'train':
            n = len(self.train_lines)
            while True:
                image_data = []
                box_data = []

                if i == 0:
                    np.random.shuffle(self.train_lines)
                for _ in range(self.batch_size):
                    if self.data_aug == 'mosaic':
                        train_data = self.train_lines[i: i+4]
                        # 防止越界
                        if i + 4 > n:
                            train_data += self.train_lines[0: (i + 4) % n]
                        image, bbox = self.get_mosaic_data(train_data)
                        i = (i + 4) % n
                    elif self.data_aug == 'mixup':
                        train_data = self.train_lines[i: i+2]
                        # 防止越界
                        if i + 2 > n:
                            train_data += self.train_lines[0: (i + 2) % n]
                        image, bbox = self.get_mixup_data(train_data)
                        i = (i + 2) % n
                    elif self.data_aug == 'random':
                        image, bbox = self.get_random_data(self.train_lines[i])
                        i = (i + 1) % n
                    else:
                        e = rand(0, 1)
                        if e <= 0.35:
                            train_data = self.train_lines[i: i+4]
                            if i + 4 > n:
                                train_data += self.train_lines[0: (i + 4) % n]
                            image, bbox = self.get_mosaic_data(train_data)
                            i = (i + 4) % n
                        elif 0.3 < e <= 0.6:
                            train_data = self.train_lines[i: i+2]
                            if i + 2 > n:
                                train_data += self.train_lines[0: (i + 2) % n]
                            image, bbox = self.get_mixup_data(train_data)
                            i = (i + 2) % n
                        else:
                            image, bbox = self.get_random_data(self.train_lines[i])
                            i = (i + 1) % n

                    image_data.append(image)
                    box_data.append(bbox)

                image_data = np.array(image_data)
                box_data = np.array(box_data)

                box_data = self.process_true_bbox(box_data)

                yield image_data, box_data

        else:
            while True:
                n = len(self.validation_lines)
                image_data = []
                box_data = []

                for _ in range(self.batch_size):
                    image, bbox = self.get_data(self.validation_lines[i])
                    i = (i + 1) % n
                    image_data.append(image)
                    box_data.append(bbox)

                image_data = np.array(image_data)
                box_data = np.array(box_data)

                box_data = self.process_true_bbox(box_data)
                yield image_data, box_data


def rand(small=0., big=1.):
    return np.random.rand() * (big - small) + small

