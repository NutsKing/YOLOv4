# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/5/22 10:39
# @Software: PyCharm
# @Brief:


import tensorflow as tf
import numpy as np
import config.config as cfg
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2 as cv


class ReadYolo4Data:
    """
    tf.data.Dataset高速读取数据，提高GPU利用率
    """

    def __init__(self, data_path, input_shape, batch_size, max_boxes=20):
        self.data_path = data_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.max_boxes = max_boxes

    def read_data_and_split_data(self):
        with open(self.data_path, "r") as f:
            files = f.readlines()

        split = int(cfg.valid_rate * len(files))
        train = files[split:]
        valid = files[:split]

        return train, valid

    def parse(self, annotation_line):
        """
        为tf.data.Dataset.map编写合适的解析函数，由于函数中某些操作不支持
        python类型的操作，所以需要用py_function转换，定义的格式如下
            Args:
              @param annotation_line: 是一行数据（图片路径 + 预测框位置）
        tf.py_function
            Args:
              第一个是要转换成tf格式的python函数，
              第二个输入的参数，
              第三个是输出的类型
        """
        if cfg.data_pretreatment == "mosaic":
            self.max_boxes *= 4
            image, bbox = tf.py_function(self.get_random_data_with_mosaic, [annotation_line], [tf.float32, tf.int32])
        elif cfg.data_pretreatment == "random":
            # 先对图片进行尺度处理，再对box位置处理成yolov3的格式
            image, bbox = tf.py_function(self._get_random_data, [annotation_line], [tf.float32, tf.int32])
        else:
            image, bbox = tf.py_function(self._get_data, [annotation_line], [tf.float32, tf.int32])

        # py_function没有解析List的返回值，所以要拆包 再合起来传出去
        y_true_13, y_true_26, y_true_52 = tf.py_function(self.process_true_bbox, [bbox],
                                                         [tf.float32, tf.float32, tf.float32])
        h, w = self.input_shape
        y_true_13.set_shape([h // 32, w // 32, 3, 25])
        y_true_26.set_shape([h // 16, w // 16, 3, 25])
        y_true_52.set_shape([h // 8, w // 8, 3, 25])
        box_data = y_true_13, y_true_26, y_true_52

        image.set_shape([h, w, 3])

        # 如果py_function的输出有个[..., ...]那么结果也会是列表，一般单个使用的时候，可以不用加[]
        return image, box_data

    def _get_data(self, annotation_line):
        """
        不对数据进行增强处理，只进行简单的尺度变换和填充处理
        :param annotation_line: 一行数据
        :return: image, box_data
        """
        line = str(annotation_line.numpy(), encoding="utf-8").split()
        image_path = line[0]
        bbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        input_height, input_width = self.input_shape

        image_height_f = tf.cast(image_height, tf.float32)
        image_width_f = tf.cast(image_width, tf.float32)
        input_height_f = tf.cast(input_height, tf.float32)
        input_width_f = tf.cast(input_width, tf.float32)

        scale = min(input_width_f / image_width_f, input_height_f / image_height_f)
        new_height = image_height_f * scale
        new_width = image_width_f * scale

        # 将图片按照固定长宽比进行缩放 空缺部分 padding
        dx_f = (input_width - new_width) / 2
        dy_f = (input_height - new_height) / 2
        dx = tf.cast(dx_f, tf.int32)
        dy = tf.cast(dy_f, tf.int32)

        # 其实这一块不是双三次线性插值resize导致像素点放大255倍，原因是：无论是cv还是plt在面对浮点数时，仅解释0-1完整比例
        image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)
        new_image = tf.image.pad_to_bounding_box(image, dy, dx, input_height, input_width)

        # 生成image.shape的大小的全1矩阵
        image_ones = tf.ones_like(image)
        image_ones_padded = tf.image.pad_to_bounding_box(image_ones, dy, dx, input_height, input_width)
        # 做个运算，白色区域变成0，填充0的区域变成1，再* 128，然后加上原图，就完成填充灰色的操作
        image = (1 - image_ones_padded) * 128 + new_image

        # 将图片归一化到0和1之间
        image /= 255.
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        # 为填充过后的图片，矫正bbox坐标
        box_data = np.zeros((self.max_boxes, 5), dtype='float32')
        if len(bbox) > 0:
            # np.random.shuffle(bbox)
            if len(bbox) > self.max_boxes:
                bbox = bbox[:self.max_boxes]

            bbox[:, [0, 2]] = bbox[:, [0, 2]] * scale + dx_f
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * scale + dy_f
            box_data[:len(bbox)] = bbox

        return image, box_data

    @staticmethod
    def merge_bboxes(bboxes, cutx, cuty):
        """
        四张图的box的合并，合并前是都是基于0坐标的Box。现在要将box合并到同一个坐标系下
        :param bboxes:
        :param cutx:
        :param cuty:
        :return:
        """
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
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
                merge_bbox.append(tmp_box)

        return merge_bbox

    @staticmethod
    def __rand(small=0., big=1.):
        return np.random.rand() * (big - small) + small

    def _get_random_data(self, annotation_line):
        """
        数据增强（改变长宽比例、大小、亮度、对比度、颜色饱和度）
        :param annotation_line: 一行数据
        :return: image, box_data
        """
        line = str(annotation_line.numpy(), encoding="utf-8").split()
        image_path = line[0]
        bbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        input_height, input_width = self.input_shape
        flip = False

        # 随机左右翻转50%
        if self.__rand(0, 1) > 0.5:
            image = tf.image.random_flip_left_right(image, seed=1)
            flip = True
        # 改变亮度，max_delta必须是float且非负数
        image = tf.image.random_brightness(image, 0.2)
        # 对比度调节
        image = tf.image.random_contrast(image, 0.3, 2.0)
        # # 色相调节
        image = tf.image.random_hue(image, 0.15)
        # 饱和度调节
        image = tf.image.random_saturation(image, 0.3, 2.0)

        # 对图像进行缩放并且进行长和宽的扭曲，改变图片的比例
        image_ratio = self.__rand(0.6, 1.4)
        # 随机生成缩放比例，缩小或者放大
        scale = self.__rand(0.3, 1.5)

        # 50%的比例改变width, 50%比例改变height
        if self.__rand(0, 1) > 0.5:
            new_height = int(scale * input_height)
            new_width = int(input_width * scale * image_ratio)
        else:
            new_width = int(scale * input_width)
            new_height = int(input_height * scale * image_ratio)

        # 这里不以scale作为判断条件是因为，尺度缩放的时候，即使尺度小于1，但图像的长宽比会导致宽比input_shape大
        # 会导致第二种条件，图像填充为黑色
        if new_height < input_height or new_width < input_width:
            new_width = input_width if new_width > input_width else new_width
            new_height = input_height if new_height > input_height else new_height

            # 将变换后的图像，转换为416x416的图像，其余部分用灰色值填充。
            # 将图片按照固定长宽比进行缩放 空缺部分 padding
            dx = tf.cast((input_width - new_width) / 2, tf.int32)
            dy = tf.cast((input_height - new_height) / 2, tf.int32)

            # 按照计算好的长宽进行resize
            image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)
            new_image = tf.image.pad_to_bounding_box(image, dy, dx, input_height, input_width)

            # 生成image.shape的大小的全1矩阵
            image_ones = tf.ones_like(image)
            image_ones_padded = tf.image.pad_to_bounding_box(image_ones, dy, dx, input_height, input_width)
            # 做个运算，白色区域变成0，填充0的区域变成1，再* 128，然后加上原图，就完成填充灰色的操作
            image = (1 - image_ones_padded) * 128 + new_image

        else:
            # 按照计算好的长宽进行resize，然后进行自动的裁剪
            image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)
            image = tf.image.resize_with_crop_or_pad(image, input_height, input_width)

        # 将图片归一化到0和1之间
        image /= 255.
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        # boxes的位置也需要修改一下
        box_data = np.zeros((self.max_boxes, 5))
        if len(bbox) > 0:

            dx = (input_width - new_width) // 2
            dy = (input_height - new_height) // 2

            bbox[:, [0, 2]] = bbox[:, [0, 2]] * new_width / image_width + dx
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * new_height / image_height + dy
            if flip:
                bbox[:, [0, 2]] = input_width - bbox[:, [2, 0]]

            # 定义边界
            bbox[:, 0:2][bbox[:, 0:2] < 0] = 0
            bbox[:, 2][bbox[:, 2] > input_width] = input_width
            bbox[:, 3][bbox[:, 3] > input_height] = input_height

            # 计算新的长宽
            box_w = bbox[:, 2] - bbox[:, 0]
            box_h = bbox[:, 3] - bbox[:, 1]
            # 去除无效数据
            bbox = bbox[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(bbox) > self.max_boxes:
                bbox = bbox[:self.max_boxes]

            box_data[:len(bbox)] = bbox

        return image, box_data, imgs

    def get_random_data_with_mosaic(self, annotation_line, hue=.1, sat=1.5, val=1.5):
        input_height, input_width = self.input_shape

        min_offset_x = 0.45
        min_offset_y = 0.45
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_datas = []
        box_datas = []

        # 定义分界线，用列表存储
        place_x = [0, 0, int(input_width * min_offset_x), int(input_width * min_offset_x)]
        place_y = [0, int(input_height * min_offset_y), int(input_width * min_offset_y), 0]
        for i in range(4):
            line = str(annotation_line[i].numpy(), encoding="utf-8").split()
            image_path = line[0]
            bbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

            # 打开图片
            image = Image.open(image_path)
            image = image.convert("RGB")
            # 图片的大小
            image_width, image_height = image.size

            # 是否翻转图片
            flip = self.__rand() < 0.5
            if flip and len(bbox) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                bbox[:, [0, 2]] = image_width - bbox[:, [2, 0]]

            # 对输入进来的图片进行缩放
            scale = self.__rand(scale_low, scale_high)
            new_height = int(scale * image_height)
            new_width = int(scale * image_width)
            image = image.resize((new_width, new_height), Image.BICUBIC)

            # 进行色域变换，hsv直接从色调、饱和度、明亮度上变化
            hue = self.__rand(-hue, hue)
            sat = self.__rand(1, sat) if self.__rand() < .5 else 1 / self.__rand(1, sat)
            val = self.__rand(1, val) if self.__rand() < .5 else 1 / self.__rand(1, val)
            x = rgb_to_hsv(np.array(image) / 255.)

            # 第一个通道是h
            x[..., 0] += hue
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            # 第二个通道是s
            x[..., 1] *= sat
            # 第三个通道是s
            x[..., 2] *= val
            x[x > 1] = 1
            x[x < 0] = 0
            image = hsv_to_rgb(x)

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[i]
            dy = place_y[i]

            mosaic_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
            mosaic_image.paste(image, (dx, dy))
            mosaic_image = np.array(mosaic_image) / 255

            # 对box进行重新处理
            if len(bbox) > 0:
                np.random.shuffle(bbox)
                # 重新计算bbox的宽高 乘上尺度 加上偏移
                bbox[:, [0, 2]] = bbox[:, [0, 2]] * new_width / image_width + dx
                bbox[:, [1, 3]] = bbox[:, [1, 3]] * new_height / image_height + dy

                # 定义边界(bbox[:, 0:2] < 0的到的是Bool型的列表，True值置为边界)
                bbox[:, 0:2][bbox[:, 0:2] < 0] = 0
                bbox[:, 2][bbox[:, 2] > input_width] = input_width
                bbox[:, 3][bbox[:, 3] > input_height] = input_height

                # 计算新的长宽
                bbox_w = bbox[:, 2] - bbox[:, 0]
                bbox_h = bbox[:, 3] - bbox[:, 1]

                # 去除无效数据
                bbox = bbox[np.logical_and(bbox_w > 1, bbox_h > 1)]
                bbox = np.array(bbox, dtype=np.float)

            image_datas.append(mosaic_image)
            box_datas.append(bbox)

        # 随机选取分界线，将图片放上去
        cutx = np.random.randint(int(input_width * min_offset_x), int(input_width * (1 - min_offset_x)))
        cuty = np.random.randint(int(input_height * min_offset_y), int(input_height * (1 - min_offset_y)))

        mosaic_image = np.zeros([input_height, input_width, 3])
        mosaic_image[:cuty, :cutx] = image_datas[0][:cuty, :cutx]
        mosaic_image[cuty:, :cutx] = image_datas[1][cuty:, :cutx]
        mosaic_image[cuty:, cutx:] = image_datas[2][cuty:, cutx:]
        mosaic_image[:cuty, cutx:] = image_datas[3][:cuty, cutx:]

        # 对框进行坐标系的处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        # 将box进行调整
        box_data = np.zeros((self.max_boxes, 5))
        if len(new_boxes) > 0:
            if len(new_boxes) > self.max_boxes:
                new_boxes = new_boxes[:self.max_boxes]
            box_data[:len(new_boxes)] = new_boxes

        return mosaic_image, box_data

    def process_true_bbox(self, box_data):
        """
        对真实框处理，首先会建立一个13x13，26x26，52x52的特征层，具体的shape是
        [b, n, n, 3, 25]的特征层，也就意味着，一个特征层最多可以存放n^2个数据
        :param box_data: 实际框的数据
        :return: 处理好后的 y_true
        """
        # [anchors[mask] for mask in anchor_masks]

        # 维度(b, max_boxes, 5)还是一样的，只是换一下类型，换成float32
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
        y_true = [np.zeros((grid_shapes[i][0], grid_shapes[i][1], cfg.num_bbox, 5 + cfg.num_classes),
                           dtype='float32') for i in range(cfg.num_bbox)]

        # 计算哪个先验框比较符合 真实框的Gw,Gh 以最高的iou作为衡量标准
        # 因为先验框数据没有坐标，只有宽高，那么现在假设所有的框的中心在（0，0），宽高除2即可。（真实框也要做一样的处理才能匹配）
        anchor_rightdown = cfg.anchors / 2.  # 网格中心为原点(即网格中心坐标为(0,0)),　计算出anchor 右下角坐标
        anchor_leftup = -anchor_rightdown  # 计算anchor 左上角坐标

        # 长宽要大于0才有效,也就是那些为了补齐到max_boxes大小的0数据无效
        # 返回一个列表，大于0的为True，小于等于0的为false
        # 选择具体一张图片，valid_mask存储的是true or false，然后只选择为true的行
        valid_mask = boxes_wh[..., 0] > 0
        # 只选择 > 0 的行
        wh = boxes_wh[valid_mask]
        wh = np.expand_dims(wh, 1)  # 在第二维度插入1个维度 (框的数量, 2) -> (框的数量, 1, 2)
        box_rightdown = wh / 2.
        box_leftup = -box_rightdown

        # 将每个真实框 与 9个先验框对比，刚刚对数据插入的维度可以理解为 每次取一个框出来shape（1,1,2）和anchors 比最大最小值
        # 所以其实可以看到源代码是有将anchors也增加一个维度，但在不给anchors增加维度也行。
        # 计算真实框和哪个先验框最契合，计算最大的交并比 作为最契合的先验框
        intersect_leftup = np.maximum(box_leftup, anchor_leftup)
        intersect_rightdown = np.minimum(box_rightdown, anchor_rightdown)
        intersect_wh = np.maximum(intersect_rightdown - intersect_leftup, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        # 计算真实框、先验框面积
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = cfg.anchors[..., 0] * cfg.anchors[..., 1]
        # 计算最大的iou
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchors = np.argmax(iou, axis=-1)

        # best_anchor是个list，label中标了几个框，他就计算出几个。
        # enumerate对他进行遍历，所以每个框都要计算合适的先验框
        for key, value in enumerate(best_anchors):
            # 遍历三次（三种类型的框 对应 三个不同大小的特征层）
            for n in range(cfg.num_bbox):
                # 如果key（最优先验框的下表）
                if value in cfg.anchor_masks[n]:
                    # 真实框的x比例 * grid_shape的长度，一般np.array都是（y,x）的格式，floor向下取整
                    # i = x * 13, i = y * 13 -- 放进特征层对应的grid里
                    i = np.floor(true_boxes[key, 0] * grid_shapes[n][1]).astype('int32')
                    j = np.floor(true_boxes[key, 1] * grid_shapes[n][0]).astype('int32')

                    # 获取 先验框（二维列表）内索引
                    k = cfg.anchor_masks[n].index(value)
                    c = true_boxes[key, 4].astype('int32')

                    # 三个大小的特征层， 逐一赋值
                    y_true[n][j, i, k, 0:4] = true_boxes[key, 0:4]
                    y_true[n][j, i, k, 4] = 1  # 置信度是1 因为含有目标
                    y_true[n][j, i, k, 5 + c] = 1  # 类别的one-hot编码，其他都为0

        return y_true

    def make_datasets(self, annotation, mode="train"):
        """
        用tf.data的方式读取数据，以提高gpu使用率
        :param annotation: 数据行[image_path, [x,y,w,h,class ...]]
        :param mode: 训练集or验证集
        :return: 数据集
        """
        # 这是GPU读取方式
        # load train dataset
        dataset = tf.data.Dataset.from_tensor_slices(annotation)
        # 如果使用mosaic数据增强的方式，要先将4个路径合成一条数据，先传入
        if cfg.data_pretreatment == "mosaic" and mode == 'train':
            dataset = dataset.repeat().batch(4)

        if mode == "train":
            # map的作用就是根据定义的 函数，对整个数据集都进行这样的操作
            # 而不用自己写一个for循环，如：可以自己定义一个归一化操作，然后用.map方法都归一化
            dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # 打乱数据，这里的shuffle的值越接近整个数据集的大小，越贴近概率分布
            # 但是电脑往往没有这么大的内存，所以适量就好
            dataset = dataset.repeat().shuffle(buffer_size=cfg.shuffle_size).batch(self.batch_size)
            # prefetch解耦了 数据产生的时间 和 数据消耗的时间
            # prefetch官方的说法是可以在gpu训练模型的同时提前预处理下一批数据
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            cfg.data_pretreatment = 'normal'
            dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.repeat().batch(self.batch_size).prefetch(self.batch_size)

        return dataset


if __name__ == '__main__':
    reader = ReadYolo4Data("../config/2012_train.txt", cfg.input_shape, cfg.batch_size)
    # train, valid = reader.read_data_and_split_data()
    # train_datasets = reader.make_datasets(train)
    # img = tf.image.convert_image_dtype(img[3], tf.uint8, saturate=True)
    # img = tf.image.encode_jpeg(img)
    # tf.io.write_file("test.jpg", img)




