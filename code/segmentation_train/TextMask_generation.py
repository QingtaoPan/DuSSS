import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from utils.modules import ResConv
from einops import rearrange
from utils.pamr import PAMR


class ExtendedInfoNCE(nn.Module):
    def __init__(self, T_init=0.07, T_learnable=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T_init))
        if not T_learnable:
            self.logit_scale.requires_grad_(False)

    def forward(self, image_emb, text_emb):
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
        labels = torch.arange(B, dtype=torch.long, device=image_emb.device)

        # [B, C]
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # cosine similarity
        logits_per_img = torch.einsum("bnc,nc->bn", image_emb, text_emb)
        logits_per_text = torch.einsum("nc,bnc->nb", text_emb, image_emb)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = F.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = F.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)

        return loss

def masked_pool(spatial_image_emb, mask, eps=1e-6):
    mask_sum = mask.sum((2, 3), keepdim=True)  # [BN11]
    weight = mask / (mask_sum + eps)
    masked_image_emb = torch.einsum("bchw,bnhw->bnc", spatial_image_emb, weight)  # [BNC]
    return masked_image_emb

class Decoder2D(nn.Module):
    def __init__(self, C, kernel_size=3, norm="none", act="relu", double=False, n_layers=2, **kwargs):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv(
                    C, C,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    upsample=True,
                    norm=norm,
                    activ=act,
                    double=double,
                    gate=True
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TextMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.model_path = '/root/data1/journal/code/Covid_pretrained_Ours/LViT/Test_session_04.06_11h19-new/models/best_model.pth'
        self.model = torch.load(self.model_path).to(self.device)
        self.img_encoder = self.model.visual_encoder
        self.text_encoder = self.model.text_encoder
        self.last_activation = nn.Sigmoid()
        self.decoder = Decoder2D(C=128).to(self.device)
        self.text_guided_loss = ExtendedInfoNCE().to(self.device)
        self.model_pairs = [[self.img_encoder, self.text_encoder],
                            ]
        self.freeze_param()


    def forward(self, unlabeled_images, texts):
        unlabeled_images = unlabeled_images.to(self.device)

        img_feat, patch_feat = self.img_encoder(unlabeled_images)
        img_emb = self.img_encoder.global_embed(img_feat)



        img_emb = F.normalize(img_emb, dim=-1)
        patch_emb = self.img_encoder.local_embed(patch_feat)
        patch_emb = F.normalize(patch_emb, dim=-1)  # [b, 196, 128]
        patch_emb = rearrange(patch_emb, "B (H W) C -> B C H W", H=14, W=14)  # [b, 128, 14, 14]
        patch_emb_dense = self.decoder(patch_emb)  # [b, 128, 56, 56]
        patch_emb_dense = F.normalize(patch_emb_dense, dim=1)  # [b, 128, 56, 56]
        # patch_emb = rearrange(patch_emb_dense, "B C H W -> B (H W) C", H=56, W=56)  # [b, 56*56, 128]

        text_feat, word_feat, word_attn_all, sents = self.text_encoder(texts, self.device)
        text_emb = self.text_encoder.global_embed(text_feat)
        text_emb = F.normalize(text_emb, dim=-1)  # [b, 128]

        ##################################################################################################################
        # text_masks_per_batch = []
        # for img_index in range(len(patch_emb)):
        #     text_mask_coarse = patch_emb[img_index] @ text_emb[img_index].unsqueeze(-1)  # [56*56, 1]<--[56*56, 128]@[128, 1]
        #     text_mask_coarse = text_mask_coarse.reshape(56, 56)  # [56, 56]<--[56*56, 1]
        #     text_masks_per_batch.append(text_mask_coarse)
        # text_masks_per_batch = torch.stack(text_masks_per_batch)
        # text_masks_per_batch = text_masks_per_batch.unsqueeze(1)  # [b, 1, 56, 56]
        # text_masks_per_batch = F.interpolate(text_masks_per_batch, scale_factor=4, mode="nearest")  # [b, 1, 224, 224]
        # text_masks_per_batch = self.last_activation(text_masks_per_batch)  # [b, 1, 224, 224]
        ##################################################################################################################
        simmap = torch.einsum("bchw,nc->bnhw", patch_emb_dense, text_emb)  # [b, b, 56, 56]
        soft_mask = torch.sigmoid(simmap)
        mask_final = self.apply_pamr(unlabeled_images, soft_mask)  # [b, b, 56, 56]
        mask_final = self.kp_branch(patch_emb_dense, text_emb, mask_final)  # [b, b, 56, 56]
        mask_final = F.interpolate(mask_final, (224, 224), mode='bilinear')  # [B, N, 224, 224]
        mask_final = mask_final.mean(dim=1).unsqueeze(1)
        text_masks_per_batch = self.last_activation(mask_final)  # [b, 1, 224, 224]
        ##################################################################################################################

        # 计算文本引导损失
        simmap = torch.einsum("bchw,nc->bnhw", patch_emb_dense, text_emb)  # [2, 2, 56, 56]
        soft_mask = torch.sigmoid(simmap)  # [2, 2, 56, 56]
        text_guided_image_emb = masked_pool(patch_emb_dense, soft_mask)  # [2, 2, 128]
        text_guided_loss = self.text_guided_loss(text_guided_image_emb, text_emb)

        return text_guided_loss, text_masks_per_batch


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


    def apply_pamr(self, image, mask):
        image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
        pamr_iter = 10
        pamr_kernel = [1, 2, 4, 8, 12, 24]
        self.pamr = PAMR(pamr_iter, pamr_kernel).to(image.device)
        self.pamr.eval()
        self.mask = self.pamr(image, mask)
        return mask

    @torch.no_grad()
    def freeze_param(self):
        for model_pair in self.model_pairs:
            for param_img, param_text in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_img.requires_grad = False  # not update by gradient
                param_text.requires_grad = False  # not update by gradient


# img = torch.rand([2, 3, 224, 224])
# text = ['i love you', 'aaaa bbb cc']
# model = TextMask()
# model(img, text)

# img = torch.rand([2, 128, 14, 14])
# model = Decoder2D(C=128)
# y = model(img)
# print(y.shape)
