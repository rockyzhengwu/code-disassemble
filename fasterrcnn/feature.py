#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: wu.zheng midday.me

import torch
import torchvision
from torch.jit.annotations import Tuple, List, Dict, Optional

from torchvision.models.detection.transform  import  GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import numpy as np
import cv2
import dataset

images, targets = dataset.load_data()
to_tensor = torchvision.transforms.ToTensor()
images = [to_tensor(image) for image in images]
targets = [{'boxes':item['boxes'], 'labels': item['labels']} for item in targets]

original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
for img in images:
    val = img.shape[-2:]
    assert len(val) == 2
    original_image_sizes.append((val[0], val[1]))


min_size=400
max_size=800
from torchvision.models.detection.transform  import  GeneralizedRCNNTransform

## transform
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
images, targets = transform(images, targets)

#################### Feature
##### resnet
# return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
# resnet = torchvision.models.resnet50(pretrained=False)
# resnet_wrap = torchvision.models._utils.IntermediateLayerGetter(resnet, return_layers)
# resnet_out = resnet_wrap(images.tensors)
# for k, v in resnet_out.items():
#   print(k, v.shape)

##### fpn
# from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
# in_channels_stage2 = resnet.inplanes // 8
# in_channels_list = [
#         in_channels_stage2,
#         in_channels_stage2 * 2,
#         in_channels_stage2 * 4,
#         in_channels_stage2 * 8,
# ]
# print(in_channels_list)
# out_channels = 256
# fpn = FeaturePyramidNetwork(
#             in_channels_list=in_channels_list,
#             out_channels=out_channels,
#             extra_blocks=LastLevelMaxPool(),
# )
# fpn_out = fpn(resnet_out)
# for k, v in fpn_out.items():
#   print(k, v.shape)
# exit(0)

print(images.tensors.shape)
backbone = resnet_fpn_backbone('resnet50', False)
features = backbone(images.tensors)
print(type(features))
for  k, v in features.items():
  print(k, v.shape)
