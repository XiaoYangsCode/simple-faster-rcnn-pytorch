from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt

'''
图片 224*224 
VGG16网络结构:
conv1_1 relu conv1_2 relu pool                  chanel:64       224*224
conv2_1 relu conv2_2 relu pool                  chanel:128      112*112
conv3_1 relu conv3_2 relu conv3_3 relu pool     chanel:256      56*56
conv4_1 relu conv4_2 relu conv4_3 relu pool     chanel:512      28*28
conv5_1 relu conv5_2 relu conv5_3 relu pool     chanel:512      14*14
fc4096 relu dropout
fc4096 relu dropout
fc1000 
softmax

VGG (
  (features): Sequential (
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU (inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU (inplace)
    (4): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU (inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU (inplace)
    (9): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU (inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU (inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU (inplace)
    (16): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU (inplace)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU (inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU (inplace)
    (23): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU (inplace)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU (inplace)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU (inplace)
    (30): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (classifier): Sequential (
    (0): Linear (25088 -> 4096)
    (1): ReLU (inplace)
    (2): Dropout (p = 0.5)
    (3): Linear (4096 -> 4096)
    (4): ReLU (inplace)
    (5): Dropout (p = 0.5)
    (6): Linear (4096 -> 1000)
  )
)

'''

def decom_vgg16():
    '''
    将VGG16拆分成：特征网络 和 分类网络
    '''
    # 可以使用 caffe预训练的模型，或者pytorch预训练的模型 pre-trained on ImageNet
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    # 前30个层，包括relu激活层，加载与训练模型vgg16的conv5_3之前的部分，去掉最后一层 pool，一共4个 pool，缩小了 16倍
    features = list(model.features)[:30]    
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]                       # 删除最后 1000 分类层
    if not opt.use_drop:                    # 删除两个 dropout
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:             # 冻结VGG16前两个 stage 不做微调
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier     # 拆分成特征网络和分类网络


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    # 提取的特征图经过了4个pooling层 缩小了 16 倍，下采样 16 倍
    feat_stride = 16  

    # n_fg_class：数据集类别，ratios：anchor宽高比，anchor_scales：缩放比例  共有9个anchor，FasterRCNN首先提出
    def __init__(self,
                 n_fg_class=20,                 
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):

        # 特征提取器，分类器 
        extractor, classifier = decom_vgg16()

        # rpn 网络输入通道，中间通道，anchor 两个变量，缩小倍数
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        ) # 返回 rpn_locs,rpn_scores,rois,roi_indices,anchor

        # VGG16 目标检测头
        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable. extracter 
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        # 一个批次里面，把图片序号拼接到最前面
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)  # None 表示增加一个维度
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        # x 是 extracter 提取后的特征，在这个工程中 
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)     # 进行拉伸，size的第一维度是box的个数。(n,c,size,size)->(n,-1)
        fc7 = self.classifier(pool)            
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        # roi_cls_locs (n, class_num*4)  roi_scores (n, class_num)
        return roi_cls_locs, roi_scores 


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
