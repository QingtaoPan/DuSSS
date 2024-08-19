# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)  # [b, 64, 224, 224]

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Embeddings(nn.Module):
    # Construct the patch, position embeddings
    def __init__(self, config, patch_size, img_size, in_channels):  # (16, 224, 64)
        super().__init__()
        img_size = _pair(img_size)  # (224, 224)
        patch_size = _pair(patch_size)  # (16, 16)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14*14
        # img_size[0]=img
        # patch_size[0]=patch_size[1] [16, 8, 4, 2]
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))  # (1, 14*14, 64)
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)) (b, 64, 14, 14)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden) (b, 14*14, 64)
        embeddings = x + self.position_embeddings  # (b, 14*14, 64)
        embeddings = self.dropout(embeddings)
        return embeddings  # (b, 14*14, 64)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  # [B, num_patches, hidden_dim]
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_patches, out_dim]
        x = self.act_layer(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):  # dim=64
        super().__init__()
        self.num_heads = num_heads  # 8
        head_dim = dim // num_heads  # 8
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # dim=64
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # (b, 14*14, 64)
        qkv = self.qkv(x)  # [b, num_patches, 3*embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # [B, num_patches, 3, num_heads, per_HeadDim]  (b, 14*14, 3, 8, 8)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, num_patches, per_HeadDim]  # (3, b, 8, 14*14, 8)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, num_patches, per_HeadDim] [4, 8, 14*14, 8/16/32/64] easy to use tensor

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, num_patches, num_patches]--[b, 8, 14*14, 8] * [b, 8, 8, 14*14] * 8**(-0.5)
        attn = attn.softmax(dim=-1)  # 每行之和为 1
        attn = self.attn_drop(attn)  # [b, 8, 14*14, 14*14]
        x = (attn @ v)  # [B, num_heads, num_patches, per_HeadDim]--[b, 8, 14*14, 14*14] @ [b, 8, 14*14, 8]
        x = x.transpose(1, 2)  # [B, num_patches, num_heads, per_HeadDim]
        x = x.reshape(B, N, C)  # [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, num_patches, embed_dim]
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,  # dim:64  num_head:8
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)  # dim=64
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)  # [b, 14*14, 64]
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # [b, 14*14, 64]
        self.norm2 = norm_layer(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)  # 64 * 4
        self.mlp = MLP(in_dim=dim, hidden_dim=self.mlp_hidden_dim, out_dim=dim)  # 64--64*4--64

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # [b, 14*14, 64]
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # [b, 14*14, 64]
        return x  # [b, 14*14, 64]


class ConvTransBN(nn.Module):  # (convolution => [BN] => ReLU)
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class VisionTransformer(nn.Module):  # Transformer-branch
    def __init__(self, config_vit, vis, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, num_classes=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(VisionTransformer, self).__init__()
        self.config_vit = config_vit
        self.vis = vis  # False
        self.embeddings = Embeddings(config=config_vit, patch_size=patch_size, img_size=img_size, in_channels=channel_num)  # (b, 14*14, 64)
        self.depth = depth  # 1
        self.dim = embed_dim  # 64
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)  # 每一行数据归一化为均值为0，方差为1
        act_layer = nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule 随机深度衰减--获取torch.linspace(0, drop_path_rate, depth)]数值
        self.Encoder_blocks = nn.Sequential(*[   # 注意力机制--输入形状=输出形状
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.CTBN = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim//2)  # 降维--/2
        self.CTBN2 = ConvTransBN(in_channels=embed_dim*2, out_channels=embed_dim)  # 降维--/2
        self.CTBN3 = ConvTransBN(in_channels=10, out_channels=196)

    def forward(self, x, skip_x, text, reconstruct=False):  # x[b, 64, 224, 224]  text[4, num_words, 64]
        if not reconstruct:
            x = self.embeddings(x)  # [b, 14*14, 64]
            if self.dim == 64:
                x = x+self.CTBN3(text)  # [b, 14*14, 64] + [b, 196, 64] = [b, 196, 64]
            x = self.Encoder_blocks(x)  # [b, 14*14, 64]
        else:
            x = self.Encoder_blocks(x)
        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x  # [b, 14*14, 64]
        elif not reconstruct:
            x = x.transpose(1, 2)  # [B, embed_dim, num_patches]  # [b, 64, 14*14]
            x = self.CTBN(x)  # [B, embed_dim//2, num_patches]  # [b, 64, 14*14]
            x = x.transpose(1, 2)  # [B, num_patches, embed_dim//2]  # [b, 14*14, 64]
            y = torch.cat([x, skip_x], dim=2)  # [B, num_patches, embed_dim]  # [b, 14*14, 128]
            return y
        elif reconstruct:
            skip_x = skip_x.transpose(1, 2)  # [b, 512, 14*14]
            skip_x = self.CTBN2(skip_x)  # [b, 256, 14*14]
            skip_x = skip_x.transpose(1, 2)  # [b, 14*14, 256]
            y = x+skip_x  # [b, 14*14, 256]
            return y


# model = VisionTransformer(config, vis=False, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
# image = np.ones([4, 256, 56, 56])
# image = torch.from_numpy(image).float()
# text = np.ones([4, 10, 256])
# text = torch.from_numpy(text).float()
# skip_x = np.ones([4, 196, 128])
# skip_x = torch.from_numpy(skip_x).float()
# pre = model(image, skip_x, text)
# print(pre.shape)
