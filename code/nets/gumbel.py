# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch


def gumbel_sigmoid(logits: torch.Tensor, tau:float = 1, hard: bool = False):  # [b, b, 224, 224], tau:1.0, hard:True
    """Samples from the Gumbel-Sigmoid distribution and optionally discretizes.

    References:
        - https://github.com/yandexdataschool/gumbel_dpg/blob/master/gumbel.py
        - https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Note:
        X - Y ~ Logistic(0,1) s.t. X, Y ~ Gumbel(0, 1).
        That is, we can implement gumbel_sigmoid using Logistic distribution.
    """
    logistic = torch.rand_like(logits)  # 该张量由区间[0,1)上均匀分布的随机数填充  [b, b, 56, 56]
    logistic = logistic.div_(1. - logistic).log_()  # ~Logistic(0,1)  logistic中逐元素-除以-(1-logistic), 取log

    gumbels = (logits + logistic) / tau  # [b, b, 56, 56]
    y_soft = gumbels.sigmoid_()  # [b, b, 56, 56]

    if hard:  # True
        # Straight through.
        y_hard = y_soft.gt(0.5).type(y_soft.dtype)  # [b, b, 56, 56]--0/1
        # gt_ break gradient flow
        #  y_hard = y_soft.gt_(0.5)  # gt_() maintain dtype, different to gt()
        ret = y_hard - y_soft.detach() + y_soft  # [b, b, 56, 56]--0/1
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret
