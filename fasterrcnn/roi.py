#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: wu.zheng midday.me
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
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads

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
num_classes = 91
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

### rpn_head
rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

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

#### ROI
box_roi_pool=None
box_head=None
box_predictor=None,
box_score_thresh=0.05
box_nms_thresh=0.5
box_detections_per_img=100
box_fg_iou_thresh=0.5
box_bg_iou_thresh=0.5
box_batch_size_per_image=512
box_positive_fraction=0.25
bbox_reg_weights=None

box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

# box_features = box_roi_pool(features, proposals, images.image_sizes)
# print(box_features.shape)

representation_size = 1024
box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

resolution = box_roi_pool.output_size[0]
box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
detections, detector_losses = roi_heads(features, proposals, images.image_sizes, targets)

print(detections)
