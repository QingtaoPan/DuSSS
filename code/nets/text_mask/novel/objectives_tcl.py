import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import numpy as np
import functools
from einops import rearrange


def Wasserstein2(mu1, sigma1, mu2, sigma2): # 2W距离，传入图片和文本的均值和标准差  [b, 768]
    bs1 = mu1.shape[0]  # b
    bs2 = mu2.shape[0]  # b
    mu1 = torch.stack([mu1]*bs2, dim=1)  # [b, b, 512]
    sigma1 = torch.stack([sigma1]*bs2, dim=1)  # [b, b, 512]
    mu2 = torch.stack([mu2]*bs1, dim=0)  # [b, b, 512]
    sigma2 = torch.stack([sigma2]*bs1, dim=0)  # [b, b, 512]
    p1 = torch.sum(torch.pow(mu1 - mu2, 2), dim=-1)  # [b, b]
    p2 = torch.sum(torch.pow(sigma1 - sigma2, 2), dim=-1)  # [b, b]
    return p1+p2, p1  # [b, b]

def compute_contrast_i2t(certain_value, ret, temp):
    image_mu = ret['image_mu'][:, 0]  # [b, 512]
    image_sigma = torch.exp(ret['image_logsigma'][:, 0])  # [b, 512]
    text_mu = ret['text_mu_aug'][:, 0]  # [b, 512]
    text_sigma = torch.exp(ret['text_logsigma_aug'][:, 0])  # [b, 512]
    bs = image_mu.shape[0]  # b
    W2_distance, mu_distance = Wasserstein2(image_mu, image_sigma, text_mu, text_sigma)   # [b, b]
    W2_distance_weighted = (-1/200 * W2_distance + 4.0) / temp  # negative_scale:1/200, shift:4, temp:0.07-->[b, b]
    similarity = W2_distance_weighted
    labels = torch.arange(bs).to(similarity.device)  # [1, b]
    loss_it = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)) / 2
    return loss_it

def compute_contrast_t2i(certain_value, ret, temp):
    image_mu = ret['image_mu_aug'][:, 0]  # [b, 512]
    image_sigma = torch.exp(ret['image_logsigma_aug'][:, 0])  # [b, 512]
    text_mu = ret['text_mu'][:, 0]  # [b, 512]
    text_sigma = torch.exp(ret['text_logsigma'][:, 0])  # [b, 512]
    bs = image_mu.shape[0]  # b
    W2_distance, mu_distance = Wasserstein2(image_mu, image_sigma, text_mu, text_sigma)   # [b, b]
    W2_distance_weighted = (-1/200 * W2_distance + 4.0) / temp  # negative_scale:1/200, shift:4, temp:0.07-->[b, b]
    similarity = W2_distance_weighted
    labels = torch.arange(bs).to(similarity.device)  # [1, b]
    loss_ti = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)) / 2
    return loss_ti

def compute_contrast_i2i(certain_value, ret, temp):
    image_mu = ret['image_mu'][:, 0]  # [b, 512]
    image_sigma = torch.exp(ret['image_logsigma'][:, 0])  # [b, 512]
    text_mu = ret['image_mu_aug'][:, 0]  # [b, 512]
    text_sigma = torch.exp(ret['image_logsigma_aug'][:, 0])  # [b, 512]
    bs = image_mu.shape[0]  # b
    W2_distance, mu_distance = Wasserstein2(image_mu, image_sigma, text_mu, text_sigma)   # [b, b]
    W2_distance_weighted = (-1/200 * W2_distance + 4.0) / temp  # negative_scale:1/200, shift:4, temp:0.07-->[b, b]
    similarity = W2_distance_weighted
    labels = torch.arange(bs).to(similarity.device)  # [1, b]
    loss_ii = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)) / 2
    return loss_ii

def compute_contrast_t2t(certain_value, ret, temp):
    image_mu = ret['text_mu'][:, 0]  # [b, 512]
    image_sigma = torch.exp(ret['text_logsigma'][:, 0])  # [b, 512]
    text_mu = ret['text_mu_aug'][:, 0]  # [b, 512]
    text_sigma = torch.exp(ret['text_logsigma_aug'][:, 0])  # [b, 512]
    bs = image_mu.shape[0]  # b
    W2_distance, mu_distance = Wasserstein2(image_mu, image_sigma, text_mu, text_sigma)   # [b, b]
    W2_distance_weighted = (-1/200 * W2_distance + 4.0) / temp  # negative_scale:1/200, shift:4, temp:0.07-->[b, b]
    similarity = W2_distance_weighted
    labels = torch.arange(bs).to(similarity.device)  # [1, b]
    loss_tt = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(0, 1), labels)) / 2
    return loss_tt

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
