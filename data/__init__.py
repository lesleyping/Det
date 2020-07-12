from .voc import VOCDetection, VOC_ROOT, VOC_CLASSES, voc

import cv2
import torch
import numpy as np

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def base_transform(self, img, size, mean):
    x = cv2.resize(img, (size, size)).astype(np.float32)
    x = x - mean
    x = x.astype(np.float32)
    return x

class Transform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def transform(self, image, boxes, labels):
        return base_transform(image, self.size, self.mean), boxes, labels
