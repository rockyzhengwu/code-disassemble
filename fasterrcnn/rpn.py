#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: wu.zheng midday.me

import torch
import torchvision
from torch.jit.annotations import Tuple, List, Dict, Optional

from torchvision.models.detection.transform  import  GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import (AnchorGenerator, RPNHead, 
    RegionProposalNetwork, concat_box_prediction_layers)

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

## feature
backbone = resnet_fpn_backbone('resnet50', False)
features = backbone(images.tensors)

########## rpn

### anchor generator
out_channels = backbone.out_channels
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
# feature_maps = list(features.values())
# for f in feature_maps:
#   print(f.shape)

# anchors = rpn_anchor_generator(images, feature_maps)
# for anchor in anchors:
#   print(anchor.shape)
# exit(0)



### rpn_head
print(rpn_anchor_generator.num_anchors_per_location())
rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
# objectness, pred_bbox_deltas = rpn_head(list(features.values()))
# for obj in objectness:
#   print(obj.shape)
# for delta in pred_bbox_deltas:
#   print(delta.shape)

### RegionProposalNetwork
# rpn other parameter
rpn_pre_nms_top_n_train=2000
rpn_pre_nms_top_n_test=1000
rpn_post_nms_top_n_train=2000
rpn_post_nms_top_n_test=1000
rpn_nms_thresh=0.7
rpn_fg_iou_thresh=0.7
rpn_bg_iou_thresh=0.3
rpn_batch_size_per_image=256
rpn_positive_fraction=0.5

## rpn head + anchor_generator + nms = regionProposalNetwork
rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
rpn_nms_thresh=0.7
rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

proposals, proposal_losses = rpn(images, features, targets)
for i, pro in enumerate(proposals):
  print("proposal %d"%(i), pro.shape)
