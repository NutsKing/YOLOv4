# Yolov4

> Are you looking for lightweight YOLOv4? Stop looking, this repository is what you want.<br>

![show.jpg](https://i.loli.net/2020/12/28/coRpvPNWUJG4ur6.jpg)

## New feature for YOLOv3

### Bag of  freebies

- [x] Mixup
- [ ] CutOut
- [ ] CutMix
- [ ] Blur
- [x] Mosaic
- [x] Focal loss
- [x] CIoU loss
- [x] Label smoothing
- [ ] DropConnect
- [x] DropBlock, DropOut

### Bag of specials

- [x] SPPNet
- [x] PANet
- [ ] SE, SAM
- [ ] SFAM, ASFF, BiFPN
- [x] Mish
- [x] CSP-Darknet53
- [ ] CBN

## 1. Quick Start.

1. Clone the repository.

```
$ git clone https://github.com/Runist/YOLOv4.git
```

2. You are supposed to install some dependencies before getting out hands with these codes.

```
$ cd YOLOv4
$ pip install -r requirements.txt
```

3. Download VOC weights.

```
$ wget https://github.com/Runist/YOLOv4/releases/download/v1.0/voc_weight.h5
```

## 2. Train your dataset.

One files are required as follows:

- [`train.txt`](https://github.com/Runist/YOLOv3/blob/main/config/train.txt): 

```
xxx/xxx.jpg 174,101,349,351,14
xxx/xxx.jpg 104,78,375,183,0 133,88,197,123,0 195,180,213,229,14 26,189,44,238,14
# image_path x_min,y_min,x_max,y_max,class_id  x_min,y_min,...,class_id 
# make sure that x_max < width and y_max < height
```

And you need edit config.py:

```python
annotation_path = "./config/train.txt"
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

Finally

```
$ mkdir logs
$ cd logs
$ mkdir model
$ python train.py			
```

## 3. Show your predict result.

```
$ python predict.py
```

If your want to change picture path, please edit predict.py:

```python
if __name__ == '__main__':
    img_path = "Your image path."
```
## 4. Reference.

- [bubbliiiing/yolo4-keras](https://github.com/bubbliiiing/yolo4-keras)

