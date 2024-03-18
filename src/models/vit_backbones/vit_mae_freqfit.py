#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from https://github.com/facebookresearch/mae/blob/main/models_vit.py
https://github.com/ryongithub/GatedPromptTuning/blob/main/src/models/vit_backbones/vit_mae.py
"""
from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.layers import Mlp, DropPath

from ..gfn import GlobalFilter

# based on timm Attention implementation
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, temp=1.0):
        """
        temp = 1.0 by default or learnable scalar
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = (attn / temp).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, temp=1.0):
        """
        temp = 1.0 by default or learnable scalar
        """
        x = x + self.drop_path(self.attn(self.norm1(x), temp=temp))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, filter_config=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool

        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.embed_dim = embed_dim

        dpr = [x.item() for x in
               torch.linspace(0, kwargs['drop_path_rate'], kwargs['depth'])]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=kwargs['num_heads'], mlp_ratio=kwargs['mlp_ratio'],
                qkv_bias=kwargs['qkv_bias'],
                drop_path=dpr[i], norm_layer=kwargs['norm_layer'])
            for i in range(kwargs['depth'])])
        # if pretrained_norm:
        self.norm = norm_layer(embed_dim)
        # self.fc_norm = norm_layer(embed_dim)

        self.filter_config = filter_config
        if self.filter_config:
            self.filter_layer = GlobalFilter(len(self.blocks) + 1, self.embed_dim, (14 ** 2) // 2 + 1)


    def _filter_ops(self, block_i, x):
        fil_in = x[:, 1:, :]    # prompt + imgs

        B, N, C = fil_in.shape
        fil_out = self.filter_layer(block_i, fil_in)    # freq filter

        # class + prompt + imgs
        x = torch.cat((x[:, 0, :].view(B, 1, C), fil_out), dim=1)

        return x

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i in range(len(self.blocks)):
            if self.filter_config:
                x = self._filter_ops(i, x)
            x = self.blocks[i](x)

        if self.filter_config:
            x = self._filter_ops(-1, x)

        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token

        outcome = self.norm(x)

        return outcome


def build_model(model_type, filter_config):
    if "vitb" in model_type:
        return vit_base_patch16(filter_config)


def vit_base_patch16(filter_config, **kwargs):
    model = VisionTransformer(
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  filter_config=filter_config,
        **kwargs)
    return model
