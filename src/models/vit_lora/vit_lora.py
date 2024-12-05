#!/usr/bin/env python3

import copy
import logging
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

# from .base_vit import VisionTransformer as ViT_ssf
from functools import partial

from os.path import join as pjoin
from turtle import forward


import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from ...configs import vit_configs as configs

from ..gfn import GlobalFilter, GlobalFilter2D

logger = logging.getLogger(__name__)

CONFIGS = {
    # "sup_vitb8": configs.get_b16_config(),
    "sup_vitb16_224": configs.get_b16_config(),
    "sup_vitb16": configs.get_b16_config(),
    "sup_vitl16_224": configs.get_l16_config(),
    "sup_vitl16": configs.get_l16_config(),
    "sup_vitb16_imagenet21k": configs.get_b16_config(),
    "sup_vitl16_imagenet21k": configs.get_l16_config(),
    "sup_vitl32_imagenet21k": configs.get_l32_config(),
    'sup_vitb32_imagenet21k': configs.get_b32_config(),
    'sup_vitb8_imagenet21k': configs.get_b8_config(),
    'sup_vith14_imagenet21k': configs.get_h14_config(),
}

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, r: int, alpha: int, dim_in: int, dim_out: int):
        super().__init__()
        self.w = w
        self.lora_a = nn.Linear(dim_in, r, bias=False)
        self.lora_b = nn.Linear(r, dim_out, bias=False)
        self.lora_r = r
        self.lora_alpha = alpha
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        x = self.w(x) + (self.lora_alpha // self.lora_r) * self.lora_b(self.lora_a(x))
        return x


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  # B, num_patches, head_size*num_head
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # B, num_head, num_patches, head_size

        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))  # B, num_head, num_patches, num_patches
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)  # B, num_head, num_patches(query), num_patches(key)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # B, num_head, num_patches, head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


def init_ssf_scale_shift(block, dim):
    '''
    https://github.com/dongzelian/SSF/blob/main/models/vision_transformer.py
    '''
    scale = nn.Parameter(torch.ones(block, dim))
    shift = nn.Parameter(torch.zeros(block, dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    '''
    https://github.com/dongzelian/SSF/blob/main/models/vision_transformer.py
    '''
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.w.weight.copy_(query_weight)
            self.attn.key.w.weight.copy_(key_weight)
            self.attn.value.w.weight.copy_(value_weight)
            self.attn.out.w.weight.copy_(out_weight)
            self.attn.query.w.bias.copy_(query_bias)
            self.attn.key.w.bias.copy_(key_bias)
            self.attn.value.w.bias.copy_(value_bias)
            self.attn.out.w.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.w.weight.copy_(mlp_weight_0)
            self.ffn.fc2.w.weight.copy_(mlp_weight_1)
            self.ffn.fc1.w.bias.copy_(mlp_bias_0)
            self.ffn.fc2.w.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, freqfit_config):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        self.freqfit_config = freqfit_config

        if self.freqfit_config == "freqfit":
            self.filter_layer = GlobalFilter2D(len(self.layer) + 1, config.hidden_size)
        elif self.freqfit_config == "ssf":
            self.ssf_scale, self.ssf_shift = init_ssf_scale_shift(config.transformer["num_layers"] + 1,
                                                                  config.hidden_size)

    def _filter_ops(self, block_i, x):
        fil_in = x[:, 1:, :]
        B, N, C = fil_in.shape
        fil_out = self.filter_layer(block_i, fil_in)
        x = torch.cat((x[:, 0, :].view(B, 1, C), fil_out), dim=1)
        return x

    def forward(self, hidden_states):
        attn_weights = []
        for i, layer_block in enumerate(self.layer):
            if self.freqfit_config == "freqfit":
                hidden_states = self._filter_ops(i, hidden_states)
            elif self.freqfit_config == "ssf":
                hidden_states = ssf_ada(hidden_states, self.ssf_scale[i], self.ssf_shift[i])
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)

        if self.freqfit_config == "freqfit":
            hidden_states = self._filter_ops(-1, hidden_states)
        elif self.freqfit_config == "ssf":
            hidden_states = ssf_ada(hidden_states, self.ssf_scale[-1], self.ssf_shift[-1])

        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward_cls_layerwise(self, hidden_states):
        # hidden_states: B, 1+n_patches, dim

        if hidden_states.size(0) != 1:
            raise ValueError('not support batch-wise cls forward yet')

        cls_embeds = []
        cls_embeds.append(hidden_states[0][0])
        for i, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if i < len(self.layer) - 1:
                cls_embeds.append(hidden_states[0][0])
        encoded = self.encoder_norm(hidden_states)
        cls_embeds.append(hidden_states[0][0])

        cls_embeds = torch.stack(cls_embeds)  # 12, dim
        return cls_embeds


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, freqfit_config):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, freqfit_config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

    def forward_cls_layerwise(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        cls_embeds = self.encoder.forward_cls_layerwise(embedding_output)
        return cls_embeds


class LoRA_ViT(nn.Module):
    def __init__(
            self, model_type, num_classes=21843, lora_cfg=None, freqfit_config=None
    ):
        super(LoRA_ViT, self).__init__()
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier
        # print("freqfit_config", freqfit_config)
        self.transformer = Transformer(config, img_size=224, vis=False, freqfit_config=freqfit_config)
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()
        print("init ViT")

        self.lora_config = lora_cfg
        self._init_lora()

    def _init_lora(self):

        r = self.lora_config.RANK
        alpha = self.lora_config.ALPHA
        assert r > 0
        assert alpha > 0
        dim = 768 #vit_model.transformer.blocks[0].attn.proj_q.in_features

        for param in self.transformer.parameters():
            param.requires_grad = False

        assign_lora_dim = partial(_LoRALayer, r=r, alpha=alpha, dim_in=dim, dim_out=dim)
        assign_lora_13dim = partial(_LoRALayer, r=r, alpha=alpha, dim_in=dim, dim_out=dim*4)
        assign_lora_31dim = partial(_LoRALayer, r=r, alpha=alpha, dim_in=dim*4, dim_out=dim)
        for layer_i in self.transformer.encoder.layer:
            layer_i.ffn.fc1 = assign_lora_13dim(layer_i.ffn.fc1)
            layer_i.ffn.fc2 = assign_lora_31dim(layer_i.ffn.fc2)
            layer_i.attn.query = assign_lora_dim(layer_i.attn.query)
            layer_i.attn.key = assign_lora_dim(layer_i.attn.key)
            layer_i.attn.value = assign_lora_dim(layer_i.attn.value)
            layer_i.attn.out = assign_lora_dim(layer_i.attn.out)

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights  # attn_weights: num_layers, B, num_head, num_patches, num_patches

    def forward_cls_layerwise(self, x):
        cls_embeds = self.transformer.forward_cls_layerwise(x)
        return cls_embeds

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
