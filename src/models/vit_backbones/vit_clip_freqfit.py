import torch
import torch.nn as nn
from ..clip import clip
from ..clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from ..gfn import GlobalFilter
_tokenizer = _Tokenizer()
from functools import partial


class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_model, filter_config):
        super().__init__()
        # HACK: Assume all is vision transformer
        self.visual = clip_model.visual
        self.viz_embed_dim = clip_model.visual.conv1.weight.shape[0]
        patch_size = self.visual.conv1.weight.shape[-1]

        self.filter_config = filter_config
        if self.filter_config:
            self.filter_layer = GlobalFilter(self.visual.transformer.layers + 1, self.viz_embed_dim, (14 ** 2) // 2 + 1)


    def _filter_ops(self, block_i, x):
        fil_in = x[:, 1:, :]    # prompt + imgs

        B, N, C = fil_in.shape
        fil_out = self.filter_layer(block_i, fil_in)    # freq filter

        # class + prompt + imgs
        x = torch.cat((x[:, 0, :].view(B, 1, C), fil_out), dim=1)

        return x

    def forward(self, x: torch.Tensor):
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        for layer_idx in range(self.visual.transformer.layers):

            if self.filter_config:
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self._filter_ops(layer_idx, x)
                x = x.permute(1, 0, 2)  # NLD -> LND

            layer = self.visual.transformer.resblocks[layer_idx]
            x = layer(x)

        if self.filter_config:
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self._filter_ops(-1, x)
            x = x.permute(1, 0, 2)  # NLD -> LND

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        return x


def build_model(model_type, clip_model, filter_config):
    if "vitb" in model_type:
        return vit_base_patch16(clip_model, filter_config)


def vit_base_patch16(clip_model, filter_config, **kwargs):
    model = CLIPImageEncoder(clip_model, filter_config, **kwargs)
    return model
