#!/usr/bin/env python3

"""
ViT-related models
Note: models return logits instead of prob
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import (
    build_vit_sup_models, build_mocov3_model, build_mae_model, build_clip_model
)
from .mlp import MLP
from ..utils import logging

logger = logging.get_logger("FreqFiT")


class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False
        
        if cfg.MODEL.TRANSFER_TYPE == "adapter":
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None

        if cfg.MODEL.TRANSFER_TYPE == "lora":
            lora_cfg = cfg.MODEL.LORA
        else:
            lora_cfg = None

        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis, lora_cfg=lora_cfg, freqfit_config=cfg.FREQFIT)
        self.cfg = cfg
        self.setup_side()
        self.setup_head(cfg)

    def setup_side(self):
        if self.cfg.MODEL.TRANSFER_TYPE != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),
                ("avgpool", m.avgpool),
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis, lora_cfg, freqfit_config):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, prompt_cfg, cfg.MODEL.MODEL_ROOT,
                adapter_cfg, load_pretrain, vis, lora_cfg, freqfit_config
        )

        if transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False
                if "ssf_scale" in k or "ssf_shift" in k or "filter_layer" in k:
                    p.requires_grad = True

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' in k or "ssf_scale" in k or "ssf_shift" in k or "filter_layer" in k:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" in k or "ssf_scale" in k or "ssf_shift" in k or "filter_layer" in k:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
                if "encoder" in k:
                    p.requires_grad = False
        
        # adapter
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" in k or "ssf_scale" in k or "ssf_shift" in k or "filter_layer" in k:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        elif transfer_type == "lora":
            for k, p in self.enc.named_parameters():
                if "lora" in k or "ssf_scale" in k or "ssf_shift" in k or "filter_layer" in k:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        elif transfer_type == "boft":
            from peft import BOFTConfig, get_peft_model
            """
            https://huggingface.co/docs/peft/main/en/conceptual_guides/oft
            """
            print(self.enc)
            config = BOFTConfig(
                boft_block_size=cfg.MODEL.BOFT.BLOCK_SIZE,
                boft_n_butterfly_factor=cfg.MODEL.BOFT.N_FACTOR,
                target_modules=["attn.query", "attn.value", "attn.key", "attn.out", "ffn.fc1", "ffn.fc2"],
                boft_dropout=0.1,
                bias="boft_only",
                modules_to_save=["classifier"],
            )

            self.enc = get_peft_model(self.enc, config)

            print(self.enc)
            for k, p in self.enc.named_parameters():
                if "ssf_scale" in k or "ssf_shift" in k or "filter_layer" in k:
                    p.requires_grad = True

        elif transfer_type == "vera":
            from peft import VeraConfig, get_peft_model
            """
            https://huggingface.co/docs/peft/en/package_reference/vera
            """

            print(self.enc)
            config = VeraConfig(
                r=cfg.MODEL.VERA.R,
                target_modules=["attn.query", "attn.value", "attn.key", "attn.out", "ffn.fc1", "ffn.fc2"],
                vera_dropout =0.1,
                bias="vera_only",
                modules_to_save=["classifier"],
            )

            self.enc = get_peft_model(self.enc, config)

            print(self.enc)
            for k, p in self.enc.named_parameters():
                if "ssf_scale" in k or "ssf_shift" in k or "filter_layer" in k:
                    p.requires_grad = True

        elif transfer_type == "fft":
            from peft import FourierFTConfig, get_peft_model
            """
            https://huggingface.co/docs/peft/en/package_reference/fourierft
            """

            print(self.enc)
            config = FourierFTConfig(
                n_frequency = cfg.MODEL.FFT.FREQ,
                scaling= cfg.MODEL.FFT.SCALE,
                target_modules=["attn.query", "attn.value", "attn.key", "attn.out", "ffn.fc1", "ffn.fc2"],
                bias="fourier_only",
                modules_to_save=["classifier"],
            )

            self.enc = get_peft_model(self.enc, config)

            print(self.enc)
            for k, p in self.enc.named_parameters():
                if "ssf_scale" in k or "ssf_shift" in k or "filter_layer" in k:
                    p.requires_grad = True

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))


    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )

    def forward(self, x, return_feature=False):
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output

        if return_feature:
            return x, x
        x = self.head(x)

        return x
    
    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class SSLViT(ViT):
    """moco-v3 and mae model."""

    def __init__(self, cfg):
        super(SSLViT, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        if "moco" in cfg.DATA.FEATURE:
            build_fn = build_mocov3_model
        elif "mae" in cfg.DATA.FEATURE:
            build_fn = build_mae_model
        elif "clip" in cfg.DATA.FEATURE:
            build_fn = build_clip_model

        self.enc, self.feat_dim = build_fn(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg=adapter_cfg, filter_cfg=cfg.MODEL.FILTER
        )

        transfer_type = cfg.MODEL.TRANSFER_TYPE
        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "fc_norm" not in k and k != "norm":  # noqa
                    p.requires_grad = False
        elif transfer_type == "partial-2":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(
                        total_layer - 2) not in k and "fc_norm" not in k and k != "norm":  # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(
                        total_layer - 2) not in k and "blocks.{}".format(
                        total_layer - 3) not in k and "blocks.{}".format(
                        total_layer - 4) not in k and "fc_norm" not in k and k != "norm":  # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed.proj.weight" not in k and "patch_embed.proj.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        # adapter
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

        for k, p in self.enc.named_parameters():
            if 'filter_layer' in k:
                p.requires_grad = True
