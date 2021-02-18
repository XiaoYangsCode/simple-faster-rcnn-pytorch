import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    # 计算回归坐标偏移量
    # 计算anchors前景概率
    # 调用 ProposalCreator和使用NMS得出2000个近似目标框的坐标
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(                    # 创建 9 个锚框的 以 cell 为中心的相对坐标（9,4）
            anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0]                        # 每个点对应着 9 个锚框
        self.feat_stride = feat_stride                              # 缩小后是原图的 1/16
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0) # 前景后景特征提取
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)   # 回归特征提取
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)  # 输出2000个roi
        normal_init(self.conv1, 0, 0.01)                            # 归一化处理
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape                      # (n, 512, H/16, W/16) H 和 W是原图的长宽
        anchor = _enumerate_shifted_anchor(         # 再9个base_anchor基础上生成 hh*ww*9 个anchor，对应到原图坐标
            np.array(self.anchor_base),             # (hh*ww*9, 4)
            self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)     # hh*ww*9 // hh*ww = 9
        h = F.relu(self.conv1(x))                   # (n, 512, hh, ww)

        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = self.loc(h)                      # (n, 9 * 4, hh, ww) 窗口回归坐标
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4) # 维度上的变化 (n, 9*hh*ww, 4)

        rpn_scores = self.score(h)                  # (n, 9 * 2, hh, ww) 背景分数特征
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()  # 维度上的变化 (n, hh, ww, 9*2)
        rpn_softmax_scores = F.softmax(
            rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)             # 第四个维度作softmax
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  # 得到所有anchor前景的分类概率 (n, hh, ww, 9)，前景的标签是 1
        rpn_fg_scores = rpn_fg_scores.view(n, -1)                       # 维度上发生变化 (n, 9*hh*ww)
        rpn_scores = rpn_scores.view(n, -1, 2)                          # 背景分数特征 (n,  hh*ww*9, 2)  

        # 对每一张图片提取 roi
        # 调用 ProposalCreator 函数， rpn_locs 维度 (batch_size, n_anchor*hh*ww, 4) rpn_fg_scores (batch_size, n_anchor*hh*ww)
        # anchor 维度 (hh*ww*9, 4)  image_size 维度为 (3, H, W) 是经过数据预处理后的
        # 计算 (H/16) x (W/16) x 9 (大概20000)个anchor 属于前景的概率，并且前 12000 个并经过 NMS 得到2000个近似目标框坐标
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),         # 位置回归
                rpn_fg_scores[i].cpu().data.numpy(),    # 前景概率
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)                 # 在第 0 维度拼接 （R, 4） 如果是 训练时输出前 2000 个
        roi_indices = np.concatenate(roi_indices, axis=0)   # (R,)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    # 根据相对坐标生成所有cell的绝对坐标
    # height width 特征图的宽高
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)                # 生成坐标网格
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),             # shift 所有cell 偏置坐标 shape (hh*ww, 4)
                      shift_y.ravel(), shift_x.ravel()), axis=1)    # stack 添加一个维度

    A = anchor_base.shape[0]                                        # 9 个anchor
    K = shift.shape[0]                                              # k = hh*ww
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))          #（1,A,4)+(K,1,4)=(K,A,4)
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)          # 最终的形状 (K*A,4)
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


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
