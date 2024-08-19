# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import Config as config
from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule
from .gumbel import gumbel_sigmoid
from .losses_fun import pos_area_loss_fun, neg_area_loss_fun
from .losses_fun import pos_area_loss_fun, neg_area_loss_fun, tclf_loss_fun, tv_loss_fun, tcli_loss_fun
from .misc import parse_losses
from .vit_MaskImage import vit_maskimage_encoder
from .decoders import GDecoder


device = torch.device(config.device)

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class ConvBatchNorm_seg(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super(ConvBatchNorm_seg, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):  # 1024, 256, 2
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)  # 1024/2 = 512
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)  # [b, 512, 28,28]
        skip_x_att = self.pixModule(skip_x)  # [b, 512, 28,28]
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension  # [b, 1024, 28,28]
        return self.nConvs(x)  # [b, 256, 28,28]


class Sim2Mask(nn.Module):
    # def __init__(self, init_w=1.0, init_b=0.0, gumbel_tau=1.0, learnable=True):  # init_w:10.0, init_b:-2.5, gumbel_tau:1.0, learnable:True
    def __init__(self, init_w=10.0, init_b=-2.5, gumbel_tau=1.0, learnable=True):
        super().__init__()
        self.init_w = init_w  # 10.0
        self.init_b = init_b  # -2.5
        self.gumbel_tau = gumbel_tau  # 1.0
        self.learnable = learnable  # True

        assert not ((init_w is None) ^ (init_b is None))
        if learnable:
            self.w = nn.Parameter(torch.full([], float(init_w)))  # tensor(10., grad_fn=<MulBackward0>)-->可学习的参数
            self.b = nn.Parameter(torch.full([], float(init_b)))  # tensor(-2.5., grad_fn=<MulBackward0>)-->可学习的参数
        else:
            self.w = init_w
            self.b = init_b

    def forward(self, x, deterministic=False):  # [b, b, 56, 56]
        device = x.device
        logits = x * self.w + self.b  # [b, b, 56, 56]

        soft_mask = torch.sigmoid(logits)  # [b, b, 56, 56]
        if deterministic:
            hard_mask = soft_mask.gt(0.5).type(logits.dtype)  # soft_mask大于0.5的设置为1，小于0.5的设置为0
        else:
            hard_mask = gumbel_sigmoid(logits, hard=True, tau=self.gumbel_tau)  # 输入:([b, b, 56, 56], True, 1.0), 输出:[b, b, 56, 56]--0/1-->大于0.5的为1，小于0.5的为0

        hard_mask = hard_mask.to(device)
        soft_mask = soft_mask.to(device)

        return hard_mask, soft_mask  # [b, b, 56, 56]--0/1, [b, b, 56, 56]--float

    def extra_repr(self):
        return f'init_w={self.init_w}, init_b={self.init_b}, learnable={self.learnable}, gumbel_tau={self.gumbel_tau}'


def get_loss(hand_mask, soft_mask, image_feat, text_feat, image, mask_image_encoder):  # hard_mask[b, b, 56, 56]-0/1, soft_mask[b, b, 56, 56]-float, image_feat[b, 512, 56, 56], text_feat[b, 512], image[b, 3, 224, 224]
    B = image.size(0)
    H, W = image_feat.shape[2:]  # 56, 56
    pos_indices = torch.arange(B, dtype=torch.long, device=image_feat.device)  # 为[0, 1, 2, 3]--batch=4为例
    pos_mask = hand_mask[torch.arange(B), pos_indices].unsqueeze(1)  # [b, 1, 56, 56]--0/1, 取b*b对角线对应的矩阵-->正样本掩码
    offdiag = torch.ones(B, B, dtype=torch.bool, device=hand_mask.device)  # [b, b*d]-->全为True
    offdiag[torch.arange(B), pos_indices] = False  # 对角线变为False
    soft_pos_mask = soft_mask[torch.arange(B), pos_indices].unsqueeze(1)  # [b, 1, 56, 56]--float, 取b*b对角线对应的矩阵
    soft_neg_mask = soft_mask.masked_select(offdiag[..., None, None]).view(B, B-1, H, W)  # [b, b-1, 56, 56]--float, 取b*b对角线以外的数据

    masks = {
        "pos": pos_mask,  # [b, 1, 56, 56]

        "soft_pos": soft_pos_mask,  # [b, 1, 56, 56]
        "soft_neg": soft_neg_mask,  # [b, b-1, 56, 56]
        "soft_all": soft_mask,  # [b, b, 56, 56]
    }

    ret = {}
    ret["mask"] = masks["soft_pos"].detach()  # [b, 1, 56, 56]--float, 取b*b对角线对应的矩阵
    ret["neg_mask"] = masks["soft_neg"].detach()  # [b, b-1 , 56, 56]--float, 取b*b对角线以外的数据

    pos_mask = masks["soft_pos"]  # [b, 1, 56, 56]--float, 取b*b对角线对应的矩阵
    neg_mask = masks["soft_neg"]  # [b, b-1, 56, 56]--float, 取b*b对角线以外的数据
    mask = masks["soft_all"]  # [b, b, 56, 56]

    if config.area_w:  # 区域级别loss
        pos_area_loss = pos_area_loss_fun(pos_mask)
        ret["area_loss"] = pos_area_loss * config.area_w

        neg_area_loss = neg_area_loss_fun(neg_mask)
        ret["neg_area_loss"] = neg_area_loss * config.area_w

    if config.tcl_w:  # 特征级别loss
        tclf_loss = tclf_loss_fun(image_feat, mask, text_feat)  # [BNC] 输入:[b, 512 56, 56], mask:[b, b, 56, 56]--float, 输出:[b, b, 512]
        ret["tclf_loss"] = tclf_loss * config.tcl_w

    if config.tv_w:
        tv_img_loss, tv_mask_loss = tv_loss_fun(image_feat, mask)  # [b, 512, 56, 56]
        ret["tv_img_loss"] = tv_img_loss * config.tv_w
        ret["tv_mask_loss"] = tv_mask_loss * config.tv_w

    if config.tcl_w:  # 图像级别loss
        tcli_loss = tcli_loss_fun(masks["pos"], image, text_feat, mask_image_encoder)
        ret["tcli_loss"] = tcli_loss * config.tcl_w

    loss = parse_losses(ret)

    return loss


class LViT(nn.Module):
    def __init__(self, config_vit, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels  # 3
        self.n_classes = n_classes  # 1
        in_channels = config_vit.base_channel  # 64
        self.inc = ConvBatchNorm(n_channels, in_channels)  # (3, 64)
        self.downVit = VisionTransformer(config_vit, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config_vit, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config_vit, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config_vit, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config_vit, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config_vit, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config_vit, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config_vit, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)  # 1024, 256
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)  # 512, 128
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)  # 256, 64
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)  # 128, 64
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.image_128_256 = ConvBatchNorm(128, 256)
        self.image_256_512 = ConvBatchNorm(256, 512)
        self.text_module_768_512 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.masker = Sim2Mask()
        self.mask_image_encoder = vit_maskimage_encoder()
        self.GD_Dec = GDecoder()


    def forward(self, x, text):
        x = x.float()  # x [4,3,224,224]
        image = x
        x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)   # [4, num_words, 512]
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)  # [4, num_words, 256]
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)  # [4, num_words, 128]
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)  # [4, num_words, 64]
        y1 = self.downVit(x1, x1, text1)  # [b, 14*14, 64]====([b, 64, 224, 224], [b, 64, 224, 224], [b, num_words, 64])
        x2 = self.down1(x1)  # [b, 128, 112, 112]
        y2 = self.downVit1(x2, y1, text2)  # [b, 14*14, 128]====([b, 128, 112, 112], [b, 14*14, 64])
        x3 = self.down2(x2)  # [b, 256, 56, 56]
        y3 = self.downVit2(x3, y2, text3)  # [b, 14*14, 256]====([b, 256, 56, 56], [b, 14*14, 128])
        x4 = self.down3(x3)  # [b, 512, 28, 28]
        y4 = self.downVit3(x4, y3, text4)  # [b, 14*14, 512]====([b, 512, 28, 28], [b, 14*14, 256])
        x5 = self.down4(x4)  # [b, 512, 14, 14]
        x_out = self.GD_Dec(x5)  # # [b, 512, 56, 56]

        # # 获取mask损失
        # ################################################################################################################
        image_feat = x_out  # [b, 512, 56, 56]
        text_feat = self.text_module_768_512(text.transpose(1, 2)).transpose(1, 2)  # [b, num_word, 512]
        text_feat = torch.einsum("bac->bc", text_feat)  # [b, 512]
        image_feat_norm = F.normalize(image_feat, dim=1)
        text_feat_norm = F.normalize(text_feat, dim=1)
        simmap = torch.einsum("bchw,nc->bnhw", image_feat_norm, text_feat_norm)
        hard_mask, soft_mask = self.masker(simmap, deterministic=False)
        # ################################################################################################################
        loss = get_loss(hard_mask, soft_mask, image_feat, text_feat, image, self.mask_image_encoder)
        # ################################################################################################################
        res_seg = self.last_activation(ConvBatchNorm_seg(in_channels=soft_mask.size(1), out_channels=1).to(device)(soft_mask.to(device)))
        res_seg = F.interpolate(res_seg, (224, 224), mode='bilinear')

        return res_seg.float().detach(), loss
