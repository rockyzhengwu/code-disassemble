#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: wu.zheng midday.me

import torch
import torchvision
from torch.jit.annotations import Tuple, List, Dict, Optional
import numpy as np
import cv2
import dataset

images, targets = dataset.load_data()
to_tensor = torchvision.transforms.ToTensor()
images = [to_tensor(image) for image in images]
targets = [{'boxes':item['boxes'], 'labels': item['labels']} for item in targets]


min_size=[800,820,900]
max_size=1333
from torchvision.models.detection.transform  import  GeneralizedRCNNTransform

## transform
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
transform.train()
print(transform.training)
images, targets = transform(images, targets)
print(images.tensors.shape)
print(images.image_sizes)
print(len(targets))
print(targets[0])