import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .novel.vit import VisionTransformer
from .novel.PDE import DisTrans
from .novel import objectives
from .novel.decoders import GDecoder
from .novel.gumbel import gumbel_sigmoid
from .novel.losses_fun import tclf_loss_fun, tcli_loss_fun
from .novel.pamr import PAMR
from functools import partial
from transformers import BertTokenizer, BertModel
from einops import rearrange
import cv2

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

        logits = x * self.w + self.b  # [b, b, 56, 56]

        soft_mask = torch.sigmoid(logits)  # [b, b, 56, 56]

        if deterministic:
            hard_mask = soft_mask.gt(0.5).type(logits.dtype)  # soft_mask大于0.5的设置为1，小于0.5的设置为0
        else:
            hard_mask = gumbel_sigmoid(logits, hard=True, tau=self.gumbel_tau)  # 输入:([b, b, 56, 56], True, 1.0), 输出:[b, b, 56, 56]--0/1-->大于0.5的为1，小于0.5的为0

        return hard_mask, soft_mask  # [b, b, 56, 56]--0/1, [b, b, 56, 56]--float


def get_loss(hand_mask, soft_mask, image_feat, text_feat, image, mask_encoder, vision_proj):  # hard_mask[b, b, 56, 56]-0/1, soft_mask[b, b, 56, 56]-float, image_feat[b, 512, 56, 56], text_feat[b, 512], image[b, 3, 224, 224]
    B = image.size(0)
    H, W = image_feat.shape[2:]  # 56, 56
    pos_indices = torch.arange(B)  # 为[0, 1, 2, 3]--batch=4为例
    pos_mask = hand_mask[torch.arange(B), pos_indices].unsqueeze(1)  # [b, 1, 56, 56]--0/1, 取b*b对角线对应的矩阵-->正样本掩码
    offdiag = torch.ones(B, B, dtype=torch.bool, device=hand_mask.device)  # [b, b*d]-->全为True
    offdiag[torch.arange(B), pos_indices] = False  # 对角线变为False
    soft_pos_mask = soft_mask[torch.arange(B), pos_indices].unsqueeze(1)  # [b, 1, 56, 56]--float, 取b*b对角线对应的矩阵
    soft_neg_mask = soft_mask.masked_select(offdiag[..., None, None]).view(B, B-1, H, W)  # [b, b-1, 56, 56]--float, 取b*b对角线以外的数据
    masks = {
        "pos": pos_mask,  # [b, 1, 56, 56] hard_mask

        "soft_pos": soft_pos_mask,  # [b, 1, 56, 56] soft_mask
        "soft_neg": soft_neg_mask,  # [b, b-1, 56, 56] soft_mask
        "soft_all": soft_mask,  # [b, b, 56, 56] soft_mask
    }

    ret = {}
    ret["mask"] = masks["soft_pos"].detach()  # [b, 1, 56, 56]--soft_mask, 取b*b对角线对应的矩阵
    ret["neg_mask"] = masks["soft_neg"].detach()  # [b, b-1 , 56, 56]--soft_mask, 取b*b对角线以外的数据

    pos_mask = masks["soft_pos"]  # [b, 1, 56, 56]--soft_mask, 取b*b对角线对应的矩阵
    neg_mask = masks["soft_neg"]  # [b, b-1, 56, 56]--soft_mask, 取b*b对角线以外的数据
    mask = masks["soft_all"]  # [b, b, 56, 56]--soft_mask

    # 特征级别loss
    tclf_loss = tclf_loss_fun(image_feat, mask, text_feat)  # [BNC] 输入:[b, 512 56, 56], mask:[b, b, 56, 56]--float, 输出:[b, b, 512]
    loss_mask_feat_text = tclf_loss * 0.1

    # 图像级别loss
    tcli_loss = tcli_loss_fun(masks["pos"], image, text_feat, mask_encoder, vision_proj)  # [b, 1, 56, 56]-hard_mask, [b, 3, 224, 224]-image, [b, 512]-text_feat
    loss_mask_img_text = tcli_loss * 0.1

    loss_mask = loss_mask_feat_text + loss_mask_img_text

    return loss_mask


bert_path = '/root/data1/lvit_semi_novel/nets/text_mask/bert_model'
# bert_path = './bert_model'
class my_novel(nn.Module):
    def __init__(self, batch_size, embed_dim=512):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.batch_size = batch_size
        self.momentum = 0.995
        self.embed_dim = embed_dim
        self.GD_Dec = GDecoder()
        self.masker = Sim2Mask()
        self.last_activation = nn.Sigmoid()
        self.mask_out = nn.Conv2d(batch_size, 1, kernel_size=(1, 1), stride=(1, 1))

        self.visual_encoder = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=1, num_heads=8,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj = nn.Linear(768, self.embed_dim)
        self.vision_proj_512 = nn.Linear(768, self.embed_dim)
        #---------------------------------------- text encodeer ---------------------------------------#
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)  # text-->token
        self.text_encoder = BertModel.from_pretrained(bert_path, return_dict=True, add_pooling_layer=False)  # token-->[b, len, 768]
        self.text_proj = nn.Linear(768, self.embed_dim)
        self.text_proj_512 = nn.Linear(768, self.embed_dim)
        #---------------------------------------- text encodeer ---------------------------------------#

        self.visual_encoder_m = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=1, num_heads=8,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj_m = nn.Linear(768, self.embed_dim)
        self.vision_proj_512_m = nn.Linear(768, self.embed_dim)
        #---------------------------------------- text encodeer ---------------------------------------#
        self.tokenizer_m = BertTokenizer.from_pretrained(bert_path)  # text-->token
        self.text_encoder_m = BertModel.from_pretrained(bert_path, return_dict=True, add_pooling_layer=False)  # token-->[b, len, 768]
        self.text_proj_m = nn.Linear(768, self.embed_dim)
        self.text_proj_512_m = nn.Linear(768, self.embed_dim)
        #---------------------------------------- text encodeer ---------------------------------------#


        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        self.copy_params()  # 切断 model_m 的梯度

        # 不确定性特征高斯建模
        self.con_img_gau_encoder = DisTrans(512, 8)
        self.con_txt_gau_encoder = DisTrans(512, 8)
        self.con_img_gau_encoder.apply(objectives.init_weights)
        self.con_txt_gau_encoder.apply(objectives.init_weights)


    def forward(self, image, image_aug, texts):

        image_embeds = self.visual_encoder(image)  # [b, 197, 768]
        image_embeds_b_197_512 = self.vision_proj_512(image_embeds)  # [b, 197, 512]
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  # [b, 512]
        text_feat_all_list = []
        for text in texts:
            text_token = self.tokenizer(text[0], return_tensors="pt")
            text_token = text_token.to(image.device)
            text_embeds = self.text_encoder(**text_token).last_hidden_state  # [b, len, 768]
            if text_embeds.shape[1] > 10:
                text_embeds = text_embeds[:, :10, :]  # [b, 10, 768]
            text_feat_all_list.append(text_embeds)
        text_feat_all_list = torch.cat(text_feat_all_list)
        text_feat_all_list = text_feat_all_list.to(image.device)
        text_feat_b_len_512 = self.text_proj_512(text_feat_all_list)  # [b, 10, 512]
        text_feat = F.normalize(self.text_proj(text_feat_all_list[:,0,:]),dim=-1)  # [b, 512]

        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image_aug)  # [b, 197, 768]
            image_embeds_b_197_512_m = self.vision_proj_512_m(image_embeds_m)  # [b, 197, 512]
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  # [b, 512]
            image_feat_m_l = F.normalize(self.vision_proj_m(image_embeds_m[:,1:,:]),dim=-1)  # [b, 196, 512]
            image_feat_m_l = self.patch_pooling(image_feat_m_l)  # [b, 16, 512]
            text_feat_all_m = []
            for text in texts:
                text_token_m = self.tokenizer_m(text, return_tensors="pt")
                text_token_m = text_token_m.to(image.device)
                text_embeds_m = self.text_encoder_m(**text_token_m).last_hidden_state  # [b, len, 768]
                if text_embeds_m.shape[1] > 10:
                    text_embeds_m = text_embeds_m[:, :10, :]  # [b, 10, 768]
                text_feat_all_m.append(text_embeds_m)
            text_feat_all_m = torch.cat(text_feat_all_m)
            text_feat_all_m = text_feat_all_m.to(image.device)
            text_feat_b_len_512_m = self.text_proj_512_m(text_feat_all_m)  # [b, 10, 512]
            text_feat_m = F.normalize(self.text_proj_m(text_feat_all_m[:,0,:]),dim=-1)  # [b, 512]
            text_feat_m_l = F.normalize(self.text_proj_m(text_feat_all_m[:,1:,:]),dim=-1)  # [b, 9, 512]

        ################----------------------计算text_mask------------------------######################################################################
        image_embeds_b_512_14_14 = rearrange(image_embeds_b_197_512[:, 1:, :], "B (H W) C -> B C H W", H=14, W=14)  # [b, 512, 14, 14]
        image_embeds_b_512_56_56 = self.GD_Dec(image_embeds_b_512_14_14)  # [b, 512, 56, 56]
        image_feat_norm = F.normalize(image_embeds_b_512_56_56, dim=1)  # [b, 512, 56, 56]
        text_feat_norm = F.normalize(text_feat, dim=1)  # [b, 512]
        simmap = torch.einsum("bchw,nc->bnhw", image_feat_norm, text_feat_norm)  # [b, b, 56, 56]
        hard_mask, soft_mask = self.masker(simmap, deterministic=False)  # [b, b, 56, 56]--0/1, [b, b, 56, 56]--float
        loss_mask = get_loss(hard_mask, soft_mask, image_feat_norm, text_feat_norm, image, self.visual_encoder_m, self.vision_proj_m)  # 计算损失

        mask_final = self.apply_pamr(image, soft_mask)  # [b, b, 56, 56]
        mask_final = self.kp_branch(image_feat_norm, text_feat_norm, mask_final)  # [b, b, 56, 56]
        mask_final = F.interpolate(mask_final, (224, 224), mode='bilinear')  # [B, N, 224, 224]
        # mask_final = self.mask_out(mask_final)  # [b, 1, 224, 224]
        mask_final = mask_final.mean(dim=1).unsqueeze(1)
        mask_final = self.last_activation(mask_final)  # [b, 1, 224, 224]


        ################----------------------高斯建模--------------------------##################
        img_mu, img_logsigma, _ = self.con_img_gau_encoder(image_embeds_b_197_512, mask=None)  # [b, 197, 512]
        txt_mu, txt_logsigma, _ = self.con_txt_gau_encoder(text_feat_b_len_512, mask=None)  # [b, 10, 512]
        img_mu_aug, img_logsigma_aug, _ = self.con_img_gau_encoder(image_embeds_b_197_512, mask=None)  # [b, 197, 512]
        txt_mu_aug, txt_logsigma_aug, _ = self.con_txt_gau_encoder(text_feat_b_len_512_m, mask=None)  # [b, 10, 512]
        ret = {
            "image_mu": img_mu,  # [b, 197, 512]
            "text_mu": txt_mu,  # [b, 10, 512]
            "image_logsigma": img_logsigma,  # [b, 197, 512]
            "text_logsigma": txt_logsigma,  # [b, 10, 512]
            "image_mu_aug": img_mu_aug,  # [b, 197, 512]
            "text_mu_aug": txt_mu_aug,  # [b, 10, 512]
            "image_logsigma_aug": img_logsigma_aug,  # [b, 197, 512]
            "text_logsigma_aug": txt_logsigma_aug,  # [b, 10, 512]
        }

        #-----------------------------image-text-loss---------------------------------#
        loss_i2t = objectives.compute_contrast_i2t(img_emb=image_feat, text_emb_aug=text_feat_m, ret=ret, temp=self.temp)
        loss_t2i = objectives.compute_contrast_t2i(text_emb=text_feat, img_emb_aug=image_feat_m, ret=ret, temp=self.temp)
        #-----------------------------image-text-loss---------------------------------#

        #--------------------------------image-image-loss-----------------------------#
        loss_i2i = objectives.compute_contrast_i2i(img_emb=image_feat, img_emb_aug=image_feat_m, ret=ret, temp=self.temp)
        loss_t2t = objectives.compute_contrast_t2t(text_emb=text_feat, text_emb_aug=text_feat_m, ret=ret, temp=self.temp)
        #---------------------------------image-image-loss----------------------------#

        #--------------------image-image-text-text-in-model-loss----------------------#
        loss_t2t_inMod_l = self.in_batch_g2l_loss(text_feat_m_l, text_feat, self.temp)  # [b, 9, 512], [b, 512] = loss
        loss_i2i_inMod_l = self.in_batch_g2l_loss(image_feat_m_l, image_feat, self.temp)  # [b, 196, 512], [b, 512] = loss
        #--------------------image-image-text-text-in-model-loss----------------------#

        loss_cons = (0.2*loss_t2t_inMod_l + 0.2*loss_i2i_inMod_l + loss_i2t + loss_t2i + loss_i2i + loss_t2t)/6

        loss_total = loss_cons + loss_mask
        return loss_total

    def patch_pooling(self, x):
        pooled_patch_length = 16
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0,3,1,2)
        c1 = b1 // int(np.sqrt(pooled_patch_length))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, pooled_patch_length, dim)
        return x

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    # jinyu: in-batch g2l loss
    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim) # (N * n_locals) * d
        m_n = m.reshape(-1, dim) # N * d

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
        u_p = torch.matmul(l, m.permute(0,2,1)).unsqueeze(2) / temp # N * n_locals * 1 * 1

        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1-temp_mask))

        u_n = torch.mm(m_n, l_n.t()) / temp
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1) # N x N x n_locals x 1

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device) # N*N*1*1
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1-temp_mask))

        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return loss

    def apply_pamr(self, image, mask):
        image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
        pamr_iter = 10
        pamr_kernel = [1, 2, 4, 8, 12, 24]
        self.pamr = PAMR(pamr_iter, pamr_kernel).to(image.device)
        self.pamr.eval()
        self.mask = self.pamr(image, mask)
        return mask

    def kp_branch(self, image_emb, text_emb, org_mask, kp_w=0.3):

        image_emb = F.normalize(image_emb, dim=1)  # BCHW
        text_emb = F.normalize(text_emb, dim=-1)  # NC

        simmap = torch.einsum("b c h w, n c -> b n h w", image_emb, text_emb)

        # kp mask
        mask = torch.sigmoid((simmap - 0.25) * 10.0)
        mask = F.interpolate(mask, org_mask.shape[2:], mode='bilinear')

        # mix
        mask = kp_w * mask + (1. - kp_w) * org_mask

        return mask


# image = torch.rand([16, 3, 224, 224])
# image_aug = torch.rand([16, 3, 224, 224])
# texts = ['a b c h w c h c h c h c h ', 'a b c h w c h c h c h c h', 'a b c h w c h c h c h c h ', 'a b c h w c h c h c h c h', 'a b c h w c h c h c h c h ', 'a b c h w c h c h c h c h', 'a b c h w c h c h c h c h ', 'a b c h w c h c h c h c h',
#          'a b c h w c h c h c h c h ', 'a b c h w c h c h c h c h', 'a b c h w c h c h c h c h ', 'a b c h w c h c h c h c h', 'a b c h w c h c h c h c h ', 'a b c h w c h c h c h c h', 'a b c h w c h c h c h c h ', 'a b c h w c h c h c h c h']
# model = my_novel(batch_size=32)
# y = model(image, image_aug, texts)
