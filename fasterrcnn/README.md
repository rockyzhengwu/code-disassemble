# FasterRCNN 代码拆解

深度学习模型感觉像是乐高积木，那我尝试和小孩子一样把乐高拆看看，如果能再拼起来会觉得这玩具就再也没意思，会想着怎么改进玩法之类的。我选 Pytorch 官方 torchvision 中的实现拆开来看看。

本文不尝试去解释 FasterRCNN 是什么的问题，也就是不会去解释太多目标检测相关的东西，只是从代码实现层，看看 torchvision 是怎么实现的。


## 宏观感受

![fasterrcnn](./imgs/fasterrcnn.png)

总体流程大致如上图，输入图像经过 ```transform``` 进行预处理，然后输入到 Feature 部分， ```backbone``` 和 ```FPN``` 可以算作特征抽取和融合的作用，但是严格的区分谁是特征抽取谁是特征融合好像界限不是很清晰，但可以理解 ```FPN ```更多贡献的是特征融合，有了特征然后分两步，第一步用 ```RPN``` 预测 Rois 然后用 ```ROI_HEAD``` 做检测和识别，最后就是后处理。

## 预处理

#### 输入

通常情况下我们的输入是通过 ```OpenCV``` 或者```PIL``` 从磁盘或内存或网络读取，转换成 Tensor 作为输入。我从 coco 数据中导出了６个样本放 data 目录下。可以尝试着看看输入数据，运行```dataset.py``` 可以可视化的查看目标检测的输入如下图，标注框在这里多为规则矩形，图中的数字是类别编号。矩形框的表示方式是四个值,左上角和右下角的坐标。

![](imgs/vis_result.png)



为了看```transform```做了什么我们先把读取输入

```python
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

```

输入有两部分，一部分是 tensor 组成的 List 一部分是 Dict 组成的 List .

#### 输出

预处理在 torchvision 中在 ```GeneralizedRCNNTransform``` 它是 ```nn.Module```的子类，所以 ```forward``` 方法就是最终调用的执行过程。主要做了两件事情， ```normalize``` 和 ```resize``` 先不去管里面怎么实现的，我直接用下面的代码看下输出是什么 

```python

## transform
from torchvision.models.detection.transform  import  GeneralizedRCNNTransform
min_size=800
max_size=1333
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
images, targets = transform(images, targets)
print(type(images))
print(type(targets))
print(targets[0])
```

输出也有两部分，一部分是 ```ImageList```对象，一部分是对应的标注 ,完整的代码可以查看 ```transform.py``` ,尝试自己去运行和产生各种输出。

```images``` 是类```ImageList```的实例具体定义如下，作用类似一个容器，把图像和对应的大小打包到一起。其中tensors 是一个 4 维向量，在这里看到的输出应该是 ```[6,3,1216,1216]``` . 

```python
 def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes
```

看完上面部分我们从 transform 的输入和输出以及需要的参数中知道了 transform 做了什么。如果只是想要调用接口到此就可以了。但是这明显还有很多问题，比如不同的图片如何做 resize 的改变的长宽比吗？那如果没改变长宽比怎么做 padding 了吗？还有```ImageList``` 的```image_sizes``` 属性存储的是图像的大小，会发现这个大小不是原始图像的大小，也不是和 ```tensos``` 的大小一样，那这是什么大小？

#### 实现细节

为了搞清楚上面的问题，来看下实现细节。```GeneralizedRCNNTransform``` 的定义需要四个参数

> min_size: 输入图像 resize 后的最小边长
>
> max_size: 输入图像 resize 后的最大边长
>
> image_mean: 每个通道对应像素的均值
>
> image_std: 每个通道像素对应的标准差

训练的时候输入的一个 batch 是一个 4 维的 tensor 这就需要每个图片都是一样大小的。为了实现这个目标```GeneralizedRCNNTransform```  先对图像做 resize 标准是resize 之后的最长边等于```max_size``` 或者最短边等于 ```min_size```  然后把对所有图片做 padding 所有resize 后的最大边长。比如 resize 后的大小是　[(887, 1333), (895, 820), (800, 1060), (1200, 900), (1196, 800), (800, 1204)] 长宽最大值分别是1200, 和 1333 但是为了能被 32整除所以padding 的目标大小为 ```(ceil(1200/32) * 32, ceil(1333/32)*32)``` 也就是 ```(1216, 1344)```  

看下面的定义会发现 ```min_size ``` 和```max_size``` 可以是一个 ```List```,  如果是 ```List ```  resize 的时候会从中随机选择一个组大和最小值，按照上面的逻辑来　resize ，这就实现了多尺度训练了。

```python

class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

```

resize 部分的就完了，记得在 resize 的同时要对标注信息做相应的操作。

normalize 部分就相对简单了, 像素减去均值然后除以标准差。

通常 transform 这一步会放到 dataset 中去做，可能还有更多事情要做，比如图像增强等等一系列操作。但最终的结果一样，各部分之间的逻辑关系还是一致的。

## Feature

#### 输入

上面讲了 transform 的输出，特征提取部分的输入是对图像来做的，所以这部分的输入就是一个四维 tensor 。对应到上面部分的输出就是　```ImageList``` 的 tensors 属性。

#### 输出

```python

## backbone
print(images.tensors.shape)
backbone = resnet_fpn_backbone('resnet50', False)
features = backbone(images.tensors)
print(type(features))

for  k, v in features.items():
  print(k, v.shape)
```

上面的代码会打印输出的具体大小，用的是默认的 ```resnet50``` 和 fpn , resnet50 部分的就不在这里介绍了，但为了仔细看下 ```fpn``` 部分的实现，所以拆成两部分 ```resnet``` 和 ```fpn``` , 下面是 resnet 的部分

```python
return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
resnet = torchvision.models.resnet50(pretrained=False)
resnet = torchvision.models._utils.IntermediateLayerGetter(resnet, return_layers)
resnet_out = resnet(images.tensors)
for k, v in resnet_out.items():
  print(k, v.shape)

# 0 torch.Size([6, 256, 152, 152])
# 1 torch.Size([6, 512, 76, 76])
# 2 torch.Size([6, 1024, 38, 38])
# 3 torch.Size([6, 2048, 19, 19])
```

resnet 输出的 featuremap 放到一个 dict 里面返回，fpn 的输入是 resnet 的输出　然后对对层的特征，从最小的层开始上采样，然后和上一层求和，这就是所谓的 ```FeaturePyramidNetwork``` 简称 ```FPN``` 

```python
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
in_channels_stage2 = resnet.inplanes // 8
in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
]
print(in_channels_list)
out_channels = 256
fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
)
fpn_out = fpn(resnet_out)
for k, v in fpn_out.items():
  print(k, v.shape)

```

到此把产生特征抽取的部分给拆成了两部分，也知道了特征抽取的输入和输出，详情可自行运行 ```feature.py``` 并任意修改.

## ＲPN 

从上面的特征部分的输出我们能知道 RPN 层的输入是什么了，但是RPN 层相对会比较复杂一些，其中会涉及到 ```anchor ``` 的概念，按照下面的代码逻辑一步步走，看每一不的输入和输出，内部的细节能放的先放，可以放到去动手实现的时候

```python

### anchor generator
out_channels = backbone.out_channels
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

### rpn_head
rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
```

大致也可以分为三部分 : ```AnchorGenerator``` 和 ```RPNHead``` 最后是 前两部分的输出作为输入生成 ```RegionProposal``` 的

### AnchorGenerator 

```AnchorGenerator ``` 的逻辑相对独立

#### 输入

输入是 images 和 ```featuers``` 的 value 部分，其实是不需要知道```feature``` 的具体值，我们只需要知道 ```feature``` 的大小就可以了，所以这里的接口定义是比较冗余的。

```python
feature_maps = list(features.values())
anchors = rpn_anchor_generator(images, feature_maps)
for anchor in anchors:
  print(anchor.shape)
#torch.Size([92355, 4])
#torch.Size([92355, 4])
#torch.Size([92355, 4])
#torch.Size([92355, 4])
#torch.Size([92355, 4])
#torch.Size([92355, 4])
```

#### 输出

看到输出是一个 ```List``` 大小同　```batch_size``` 　要理解了这里为什么是　```[92355, 4]``` 的大小就差不多理解了 anchor 的生成。可以运行　```rpn.py``` 查看细节。这里简单算一下。```AnchorGEnerator``` 的参数如下, 意味着在 feature 的每个点上会有 3 个 不同的 anchor. 注意这里的大小是对应原图的，或者说是对应到 resize 之后的图上的。

```python
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
```

输入的 feature 的大小如下, 一共是 4 个 tensor ，但是生成的 anchor 是对应到每个样本的。``` （152 * 152 + 76 * 76 + 38 * 38 + 19 * 19 + 10 * 10）×3 = 92355 ```  每个点生成三个长宽比分别为 ```0.5,1.0,2.0```  根据特征的大小就有 ```92355``` 个anchor. 

```python
torch.Size([6, 256, 152, 152])
torch.Size([6, 256, 76, 76])
torch.Size([6, 256, 38, 38])
torch.Size([6, 256, 19, 19])
torch.Size([6, 256, 10, 10])
```

#### RPN Head

RPN 会为每个 Anchor 预测一个坐标，和对应是否是前景的得分。这就是 RPN_Head 做的事情，

```python
print(rpn_anchor_generator.num_anchors_per_location()[0])
rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
```

#### 输入

输入很明了就是5 个 tensor 组成的 featuer 

```python
print(rpn_anchor_generator.num_anchors_per_location())
rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
objectness, pred_bbox_deltas = rpn_head(list(features.values()))
for obj in objectness:
  print(obj.shape)
for delta in pred_bbox_deltas:
  print(delta.shape)
exit(0)
```

#### 输出

输出的 objectness 每个 anchor 是前景的得分 ，每张图对应一个样本，

```
torch.Size([6, 3, 152, 152])
torch.Size([6, 3, 76, 76])
torch.Size([6, 3, 38, 38])
torch.Size([6, 3, 19, 19])
torch.Size([6, 3, 10, 10])
```

pred_bbox_deltas 是对每个 anchor 坐标偏移量的预测，由于知道 anchor 在特征上的位置，也知道每个特征图的缩放比(stride) 就能知道 每个 anchor 在原图上的坐标，根据偏移量对anchor 做调整就能得到最后的目标检测的结果，这是 FasterRCNN 目标检测的基本逻辑。

### RegionProposalNetwork

上面生成了 Anchor 和　对每个 Anchor 都预测了对应的得分和位置偏移。在 FasterRCNN 中 RPN 并不产生最终的检测结果,只产生 ```RegionProposal``` 所以接下来就是根据 ```rpn_head``` 的预测结果生成 ```RegionProposal``` 了。　但是在训练的时候还需要根据样本来产生正负样本，并计算 RPN 层的 loss 。

```python
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
```

```RegionProposalNetwork``` 是把 anchor_generator 和 rpn_head 作为参数的。所以其实完整的逻辑在 ```RegionProposalNetwork``` 中 实现，上面已经拆分了两部分，剩下的就是生成 ```RegionProposal``` 的部分了。

```python
proposals, proposal_losses = rpn(images, features, targets)
for i, pro in enumerate(proposals):
  print("proposal %d"%(i), pro.shape)
#proposal 0 torch.Size([2000, 4])
#proposal 1 torch.Size([2000, 4])
#proposal 2 torch.Size([2000, 4])
#proposal 3 torch.Size([2000, 4])
#proposal 4 torch.Size([2000, 4])
#proposal 5 torch.Size([2000, 4])
```

rpn 最终的输出是每张图 2000 个 box 并在训练的时候计算相应的 loss . 这2000 个 proposal 会输入到 ```ROI_Head``` 中预测最终的目标检测结果。

这里有太多细节，什么样的策略从　```923555``` 个 anchor 中选择 2000 个？ 简单说就是利用 ```NMS``` 。在 ```NMS``` 之前为了减少运算量会先把选一部分出来，作为 NMS 的输入，其他的直接扔掉。这也就是 ```rpn_pre_nms_top_n_train=2000 rpn_pre_nms_top_n_test=1000``` 这两个参数的作用，一个 用在 ```train``` 一个用在```test``` 阶段。 相应的```rpn_post_nms_top_n_train``` 就是要保留的，所以最后每张图产生的 proposal　是 2000 个。　

剩下还有些比较重要但又相对有点儿复杂的问题，anchor 的坐标对应到真是图像的上，以及生成训练的样本，也就是要给参与计算 loss 的 anchor 打标签等等。

这里知道了　RPN 的作用，对理解它的参数和使用应该不是问题，剩下没有详细的问题暂时放下，比如```nms``` 的细节问题就能写上好几篇 bolg 或许能解释清楚。建议直接读代码，实现一个出来。

#### ROIHead

有了 Proposal 和特征 ROIHead 要做的是根据二者预测最终的目标检测的结果。可以从下面的代码看到可分三部分，```MultiScaleRoiAlign``` ,```FastRCNNPredictor``` ```TwoMlPHead```  

### MultiScaleRoiAlign

由于 proposal 是不同大小的，为了得到同样大小的特征，有很多方法，torchvision 默认使用的是 RoiAlign , 

```python
box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

box_features = box_roi_pool(features, proposals, images.image_sizes)
print(box_features.shape)
# torch.Size([12000, 256, 7, 7])

```

#### 输入

输入是 ``` features``` 和 ```proposals``` 还有图像大小. 

#### 输出

从输出的大小就能看到，每张图片对应的 2000 个 proposal 抽取了一个 [256, 7, 7] 的feature 有了这个 feature 就可以完成后续的预测得分以及位置信息。



ROIHead 剩下的和 RPNHead 比较相似了，

```python

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

```









