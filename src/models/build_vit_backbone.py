#!/usr/bin/env python3
import numpy as np
import torch
import os

from .vit_backbones.vit import VisionTransformer
from .vit_backbones.vit_moco import vit_base
from .vit_backbones.vit_mae import build_model as mae_vit_model

from .vit_prompt.vit import PromptedVisionTransformer
from .vit_prompt.vit_moco import vit_base as prompt_vit_base
from .vit_prompt.vit_mae import build_model as prompt_mae_vit_model
from .vit_prompt.vit_clip import build_model as prompt_clip_vit_model

from .vit_adapter.vit_mae import build_model as adapter_mae_vit_model
from .vit_adapter.vit_moco import vit_base_freqfit as adapter_vit_base
from .vit_adapter.vit_clip import build_model as adapter_clip_vit_model
from .vit_adapter.vit import ADPT_VisionTransformer

from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .vit_lora.vit_lora import LoRA_ViT

MODEL_ZOO = {
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "mae_vitb16": "mae_pretrain_vit_base.pth",
    "mocov3_vitb16" : "vit-b-300ep.pth.tar",
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

def build_vit_sup_models(
    model_type, crop_size, prompt_cfg=None, model_root=None, adapter_cfg=None,
        load_pretrain=True, vis=False, lora_cfg=None, freqfit_config=None
):
    # image size is the size of actual image
    m2featdim = {
        "sup_vitb16_224": 768,
        "sup_vitb16": 768,
        "sup_vitl16_224": 1024,
        "sup_vitl16": 1024,
        "sup_vitb8_imagenet21k": 768,
        "sup_vitb16_imagenet21k": 768,
        "sup_vitb32_imagenet21k": 768,
        "sup_vitl16_imagenet21k": 1024,
        "sup_vitl32_imagenet21k": 1024,
        "sup_vith14_imagenet21k": 1280,
    }
    if prompt_cfg is not None:
        model = PromptedVisionTransformer(
            prompt_cfg, model_type,
            crop_size, num_classes=-1, vis=vis, freqfit_config=freqfit_config
        )
    elif adapter_cfg is not None:
        model = ADPT_VisionTransformer(model_type, crop_size, num_classes=-1, adapter_cfg=adapter_cfg, freqfit_config=freqfit_config)

    elif lora_cfg is not None:
        model = LoRA_ViT(model_type=model_type, num_classes=-1, lora_cfg=lora_cfg, freqfit_config=freqfit_config)
    else:
        print("build ViT linear")
        model = VisionTransformer(
            model_type, crop_size, num_classes=-1, vis=vis, freqfit_config=freqfit_config)
    
    if load_pretrain:
        model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]
