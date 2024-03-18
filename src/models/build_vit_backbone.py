#!/usr/bin/env python3
import numpy as np
import torch
import os

from .vit_backbones.vit_mae_freqfit import build_model as mae_vit_model_freqfit
from .vit_backbones.vit_moco_freqfit import vit_base_freqfit
from .vit_backbones.vit_clip_freqfit import build_model as clip_vit_model_freqfit

from .vit_prompt.vit_moco import vit_base as prompt_vit_base
from .vit_prompt.vit_mae import build_model as prompt_mae_vit_model
from .vit_prompt.vit_clip import build_model as prompt_clip_vit_model

from .vit_adapter.vit_mae import build_model as adapter_mae_vit_model
from .vit_adapter.vit_moco import vit_base_freqfit as adapter_vit_base_freqfit

from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

MODEL_ZOO = {
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mocov3_vitb16" : "vit-b-300ep.pth.tar",    # pre-trained SSL weights
    "clip_vitb16": "ViT-B-16.pt",
}


def build_mae_model(
    model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None,
    filter_cfg=None
):
    if prompt_cfg is not None:
        model = prompt_mae_vit_model(model_type, prompt_cfg)
    elif adapter_cfg is not None:
        model = adapter_mae_vit_model(model_type, adapter_cfg)
    else:
        model = mae_vit_model_freqfit(model_type, filter_cfg)
    out_dim = model.embed_dim

    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['model']

    msg = model.load_state_dict(state_dict, strict=False)
    print('Loading pretrained weight from MAE:\n', msg)
    model.head = torch.nn.Identity()
    return model, out_dim


def build_mocov3_model(
    model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None,
    filter_cfg=None
):

    if "mocov3_vitb16" not in model_type:
        raise ValueError("Does not support other arch")
    if prompt_cfg is not None:
        model = prompt_vit_base(prompt_cfg)
    elif adapter_cfg is not None:
        model = adapter_vit_base_freqfit(adapter_cfg)
    else:
        model = vit_base_freqfit(filter_cfg)
    out_dim = 768
    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            key = k.replace('module.', '')
            if key.startswith('base_encoder.'):
                key = key.replace('base_encoder.', '')
            elif key.startswith('momentum'):
                del state_dict[k]
                continue
            state_dict[key] = state_dict[k]
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print('Loading pretrained weight from MoCo-v3:\n', msg)
    model.head = torch.nn.Identity()
    return model, out_dim

def build_clip_model(
    model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None,
    filter_cfg=None
):
    url = clip._MODELS["ViT-B/16"]
    model_path = clip._download(url, model_root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model_clip = clip.build_model(state_dict or model.state_dict())

    if prompt_cfg is not None:
        model = prompt_clip_vit_model(model_type, model_clip, prompt_cfg)
    else:
        model = clip_vit_model_freqfit(model_type, model_clip, filter_cfg)

    model.float()
    return model, model.viz_embed_dim


