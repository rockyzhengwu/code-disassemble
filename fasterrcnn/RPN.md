# RPN 实现细节
从上篇文章中从宏观的的角度知道了 FasterRCNN 各个模块的输入和输出，这样也就能理解了各模块的作用，以及彼此之间的组织关系
这里记录些 RPN(RegionProposalNetwork) 在代码实现层的一些小细节，比如下面这些问题，并通过可视化中间结果直观感受。

1. Anchor 的生成方法
2. 如何选择 anchor 做为 proposals
3. loss 的计算， 在计算 loss 之前需要从 anchor 中选择正负样本

先从两幅图看下 RPN 学到了什么，左边是 RPN 没有加载训练好的权重输出的前 100 个 proposals,第二幅是加载训练好的权重后生成 proposals， 之所以只选 100 个是为了看得清楚,　可以看到第二幅图找到的 proposals 明显质量要高很多，第一幅可能都是随机的，因为没有训练权重都随机。从右边也能感觉到 RPN 本身就可以做为目标检测的 Head

<center class="half">
<img src='fasterrcnn/imgs/random_proposal.png' style="zoom:40%">
<img src='fasterrcnn/imgs/predict_proposal.png' style="zoom:40%">
</center>

## Anchor 生成

上图这样的 proposal 是如何产生的了，是通过 ```RPNHead``` 对 Anchor 的预测得分和位置回归得到的。Anchor 是在 feature_map 的 每一个位置生成多个不同大小不同长宽比的矩形框。而且对于不同层的 feature_map 他们的感受野是不一样的，所以设置的 anchor 的大小也不一样　比如下面的参数定义了在五层不同大小的feature_map 上生成的 anchor 大小分别为 32,64,256,512。 这里是对应到输入图像大小上的边长。由于 anchor 的生成是提前定义的，所以相当于超参数一样，所以也有些方法来改进 这里的anchor 的生成方法

```python
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0)) 
```

Anchor 的生成方法一般分两步，第一步是在一个点上生成对应的 anchor 比如上面的参数就是在 不同的 feature_map 的原点生成三个 anchor 然后根据 feature_map 的大小移动这些 anchor 。可以看到生成多少 anchor 和输入图像的大小是相关的。所以这样动态生成就可以支持不同尺度的图像同时训练。搞懂 anchor 的生成过程最麻烦的可能是理解如何把生成的 anchor 在整个 feature_map 生移动，这里我理解搞懂一个函数就可以了 ```torch.meshgrid``` 对应的 numpy 里面也有相应的函数。可以根据横轴坐标和纵轴坐标得到所有的交点

##  Proposals 的选择

PRNHead 会预测在 feanture_map 的每个点上给每一个 anchor 预测一个前景得分。同时还会预测对应的位置。Proposal 的选择有两步，第一步是在每一层的 feature_map 上选择一定数量得分最高的 anchor, 然后对所有的选择做 ```nms``` . nms 的结果选择前 n 个作为最终的 proposal 

逻辑相对简单，但是比较麻烦的是在 pytorch 里面 tensor 的计算效率相对较高，这里相当于要遍历每张图的每个 feature_map ，为了尽量能用 tensor 计算 torchvision 的实现看起来就会比较费解，不过可以从[这里](https://github.com/rockyzhengwu/code-disassemble/tree/master/fasterrcnn)的代码 一行行 debug, 代码是从 torchvision 摘出来的，可以在任何地方 ```print``` 来查看运行结果。一个例子可以在 [fasterrcnn_example](https://github.com/rockyzhengwu/code-disassemble/blob/master/fasterrcnn/fasterrcnn_example.py) 中找到。

还有就是一些特殊情况的处理，比如对超出图像边缘的 anchor 做剪切等。

### loss 的计算

要计算 loss 就需要有标注数据， 这里的预测对象是 anchor 的得分和位置回归，我们有的是真是目标的位置和标签。所以 fasterrcnn 定了一个策略(规则)来对所有的 anchor 打上标签。策略和 iou 相关，这里的可以简单看下 iou 的计算方法，其想法不是去先判断连个 box 是否相交，而是假设他们相交，尝试去计算相交区域的左上角和右下角，当这个区域有边小于零的时候表示他们不想交区域为 0

```python
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

```

最终选择的正样本我们也可以画出来看一下

<img src='fasterrcnn/imgs/positive_box.png' >



有了样本还需要计算回归的目标参数，然后计算剩下的就是 计算 loss 了，loss 的计算和 proposals 的生成是没有直接相关的的，通过 loss 的反向传播来修改得分来得到更好的 proposal 所以很多计算都只是在训练过程中用到，比如对 anchor 打上标签的操作。