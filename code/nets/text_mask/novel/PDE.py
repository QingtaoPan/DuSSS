import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,  # 768
        hidden_features=None,  # 768*4
        out_features=None,
        act_layer=nn.GELU,  # nn.GELU
        drop=0.0,  # 0.1
    ):
        super().__init__()
        out_features = out_features or in_features  # 768
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # (768, 768*4)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # (768*4, 768)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  # [b, 197, 768*4]
        x = self.act(x)  # [b, 197, 768*4]
        x = self.drop(x)  # [b, 197, 768*4]
        x = self.fc2(x)  # [b, 197, 768]
        x = self.drop(x)  # [b, 197, 768]
        return x  # [b, 197, 768]


class DisAttention(nn.Module):
    def __init__(
        self,
        dim,  # 768
        num_heads=8,  # 12
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,  # 0.1
        proj_drop=0.0,  # 0.1
    ):
        super().__init__()
        self.num_heads = num_heads  # 12
        head_dim = dim // num_heads  # 64
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5  # 0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # (768, 768*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mu_proj = nn.Linear(int(dim/2), dim)  # (384, 768)
        self.mu_proj_drop = nn.Dropout(proj_drop)
        self.logsig_proj = nn.Linear(int(dim/2), dim)  # (384, 768)
        self.logsig_proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape  # [b, 197, 512]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)  # (b, 197, 3, 12, 42)
            .permute(2, 0, 3, 1, 4)
        ) # (3, B, mu_heads_num+logsig_heads_num, n, dim_heads) = [b, 197, 768]->[b, 197, 768*3]->[b, 197, 3, 12, 64]->[3, b, 12, 197, 64]
        q, k, v = (
            qkv[0],  # [b, 12, 197, 64]
            qkv[1],  # [b, 12, 197, 64]
            qkv[2],  # [b, 12, 197, 64]
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, 12, 197, 197]
        if mask is not None:
            attn = attn + mask  # [b, 12, 197, 197]
        attn = attn.softmax(dim=-1)  # [b, 12, 197, 197]
        attn = self.attn_drop(attn)  # [b, 12, 197, 197]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).reshape(B, N, 2, int(C/2))  # [b, 12, 197, 197]->[b, 12, 197, 64]->[b, 197, 12, 64]->[b, 197, 768]->[b, 197, 2, 384]

        mu = x[:,:,0,:]  # [b, 197, 384]
        logsigma = x[:,:,1,:]  # [b, 197, 384]
        mu = self.mu_proj(mu)  # [b, 197, 768]
        mu = self.mu_proj_drop(mu)  # [b, 197, 768]
        logsigma = self.logsig_proj(logsigma)  # [b, 197, 768]
        logsigma = self.logsig_proj_drop(logsigma)  # [b, 197, 768]
        return mu, logsigma, attn  # [b, 197, 768], [b, 197, 768], [b, 12, 197, 197]


class DisTrans(nn.Module):
    def __init__(
        self,
        dim,  # 512
        num_heads,  # 12
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.fc = nn.Linear(dim, dim)  # [768, 768]
        self.act = act_layer()
        self.norm1 = norm_layer(dim)
        self.attn = DisAttention(
            dim,  # 768
            num_heads=num_heads,  # 12
            qkv_bias=qkv_bias,  # False
            qk_scale=qk_scale,  # None
            attn_drop=attn_drop,  # 0.1
            proj_drop=drop,  # 0.1
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 768*4
        self.mu_mlp = Mlp(
            in_features=dim,  # 768
            hidden_features=mlp_hidden_dim,  # 768*4
            act_layer=act_layer,  # nn.GELU
            drop=drop,  # 0.1
        )
        self.logsig_mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):  # x[b, 197, 768]  ,mask[b, 197, 768]
        x_ = self.norm1(self.act(self.fc(x)))  # [b, 197, 768]
        mu, logsigma, attn = self.attn(x_, mask=mask)  # [b, 197, 768], [b, 197, 768], [b, 12, 197, 197]
        mu = x + self.drop_path(mu)  # [b, 197, 768]
        mu = mu + self.drop_path(self.mu_mlp(self.norm2(mu)))  # [b, 197, 768]
        logsigma = logsigma + self.drop_path(self.logsig_mlp(self.norm3(logsigma)))  # [b, 197, 768]
        return mu, logsigma, attn  # [b, 197, 768], [b, 197, 768], [b, 12, 197, 197]
