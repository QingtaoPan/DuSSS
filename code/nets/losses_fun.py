import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit_MaskImage import vit_maskimage_encoder
import Config as config


class AreaTCLLoss:
    def __init__(self, prior: float):
        self.prior = prior

    def __call__(self, mask: torch.Tensor):
        return (mask.mean() - self.prior).abs()


def masked_pool(spatial_image_emb, mask, eps=1e-6):  # spatial_image_emb:[b, 512 56, 56], mask:[b, b, 56, 56]--float
    """Average pool spatial_image_emb with mask

    Args:
        spatial_image_emb [BCHW]: spatial embedding
        mask [BNHW]: hard or soft mask

    Return:
        image_emb [BNC] : mask-pooled tensor
    """
    mask_sum = mask.sum((2,3), keepdim=True)  # [BN11] [b, b, 1, 1]
    weight = mask / (mask_sum + eps)  # [b, b, 56, 56]
    masked_image_emb = torch.einsum("bchw,bnhw->bnc", spatial_image_emb, weight)  # [BNC] [b, b, 512]

    return masked_image_emb  # [b, b, 512]


class InfoNCE(nn.Module):
    def __init__(self, T_init=0.07, T_learnable=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T_init))
        if not T_learnable:
            self.logit_scale.requires_grad_(False)

    def forward(self, image_emb, text_emb):  # [b, 512] [b, 512]
        """
        Args:
            image_emb [B, C]: image embedding
            text_emb [B, C]: text embedding
        """
        assert image_emb.ndim == text_emb.ndim == 2

        B = image_emb.shape[0]  # b
        # get label globally
        labels = torch.arange(B, dtype=torch.long, device=image_emb.device)  # [0, 1,...,b]

        # [B, C]
        image_emb = F.normalize(image_emb, dim=-1)  # [b, 512]
        text_emb = F.normalize(text_emb, dim=-1)  # [b, 512]

        # cosine similarity
        text_emb = text_emb.to(image_emb.device)
        logits_per_img = image_emb @ text_emb.t()  # [b, b]
        logits_per_text = text_emb @ image_emb.t()  # [b, b]

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)  # 将tensor限制为[min, max]之间
        loss_img = F.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = F.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)

        return loss


class ExtendedInfoNCE(nn.Module):
    def __init__(self, T_init=0.07, T_learnable=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T_init))
        if not T_learnable:
            self.logit_scale.requires_grad_(False)

    def forward(self, image_emb, text_emb):  # [b, b, 512], [b, 512]
        """ExtendedInfoNCE is an InfoNCE function but computes similarity map differently.

        Note:
            InfoNCE: s = einsum("ic,jc->ij", img_emb, txt_emb)
            ExtendedInfoNCE: s = einsum("ijc,jc->ij", img_emb, txt_emb)

            In practice, the implementation of ExtendedInfoNCE becomes rather complicated
            when using multi-gpu with DDP.

        Args:
            image_emb [B, N, C]: extended image embedding where N=B*D
            text_emb [B, C]: text embedding
        """
        B = image_emb.shape[0]
        # get label globally
        labels = torch.arange(B, dtype=torch.long, device=image_emb.device)  # 获取标签

        # [B, C]
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # cosine similarity
        image_emb = image_emb.float()
        text_emb = text_emb.float()
        logits_per_img = torch.einsum("bnc,nc->bn", image_emb, text_emb)  # [b, b]
        logits_per_text = torch.einsum("nbc,bc->bn", image_emb, text_emb)  # [b, b]

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = F.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = F.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)

        return loss


def tv_loss(x):
    """Total variation loss

    Args:
        x: 4-d tensor [*, *, H, W]
    """
    return (
        (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() +
        (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
    )


# 正样本区域损失
def pos_area_loss_fun(pos_mask):
    pos_area_loss = AreaTCLLoss(0.4)(pos_mask)
    return pos_area_loss


# 负样本区域损失
def neg_area_loss_fun(neg_mask):
    pos_area_loss = AreaTCLLoss(0.0)(neg_mask)
    return pos_area_loss


# 特征级损失
def tclf_loss_fun(s1_image_emb, mask, s1_text_emb):
    image_emb = masked_pool(s1_image_emb, mask)
    tclf_loss = ExtendedInfoNCE()(image_emb, s1_text_emb)
    return tclf_loss


def tv_loss_fun(s1_image_emb, mask):
    tv_img_loss = tv_loss(s1_image_emb)
    tv_mask_loss = tv_loss(mask)
    return tv_img_loss, tv_mask_loss


# 图像级别损失-输入数据类型为half()
def tcli_loss_fun(mask_pos, image, text_emb, mask_image_encoder):
    device = torch.device(config.device)
    pos_mask = F.interpolate(mask_pos, size=image.shape[2:])  # 输入-->mask["pos"]:硬标签[b, 1, 56, 56],  输出-->[b, 1, 224, 224]
    # vit_maskimage_encoder_model = vit_maskimage_encoder()
    # vit_maskimage_encoder_model = vit_maskimage_encoder_model.to(device)
    # vit_maskimage_encoder_model = nn.DataParallel(vit_maskimage_encoder_model, device_ids=[0, 1, 2, 3])
    mask_image = pos_mask*image
    masked_img_emb = mask_image_encoder(mask_image)  # 输入:pos_mask*image[b, 3, 224, 224],  输出:[b, 512]
    tcli_loss = InfoNCE()(masked_img_emb, text_emb)  # 输入:[b, 512], [b, 512]
    return tcli_loss







