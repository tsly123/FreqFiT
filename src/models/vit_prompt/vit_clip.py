import torch
import torch.nn as nn
from ..clip import clip
from ..clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.nn import Conv2d, Dropout
import math
from functools import partial, reduce
from operator import mul
from ..gfn import GlobalFilter

_tokenizer = _Tokenizer()

class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_model, prompt_config):
        super().__init__()
        # HACK: Assume all is vision transformer
        self.visual = clip_model.visual
        self.viz_embed_dim = clip_model.visual.conv1.weight.shape[0]
        patch_size = [16, 16]

        self.prompt_config = prompt_config
        if self.prompt_config.DEEP and self.prompt_config.LOCATION not in ["prepend", ]:
            raise ValueError("Deep-{} is not supported".format(self.prompt_config.LOCATION))

        num_tokens = self.prompt_config.NUM_TOKENS

        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.viz_embed_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.viz_embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    self.visual.transformer.layers - 1,
                    num_tokens, self.viz_embed_dim
                ))
                # xavier_uniform initialization
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

        if self.prompt_config.FILTER:
            self.filter_layer = GlobalFilter(self.visual.transformer.layers+1, self.viz_embed_dim, (14**2+self.num_tokens)//2 +1)


    def _filter_ops(self, block_i, x):
        fil_in = x[:, 1:, :]    # prompt + imgs

        B, N, C = fil_in.shape
        fil_out = self.filter_layer(block_i, fil_in)    # frequency filtering

        x = torch.cat((x[:, 0, :].view(B, 1, C), fil_out), dim=1)   # class + prompt + imgs
        return x

    def embeddings(self, x):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        return x


    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        if self.prompt_config.LOCATION == "prepend":
            # after CLS token, all before image patches
            x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.prompt_embeddings.expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        else:
            raise ValueError("Other prompt locations are not supported")
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.visual.transformer.resblocks.eval()
            self.visual.conv1.eval()
            self.visual.ln_pre.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    # def forward_features(self, x):
    def forward(self, x):
        x = self.incorporate_prompt(x)

        B = x.shape[0]
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(self.visual.transformer.layers):
            layer = self.visual.transformer.resblocks[i]
            if i == 0:
                if self.prompt_config.FILTER:
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self._filter_ops(i, x)
                    x = x.permute(1, 0, 2)  # NLD -> LND

                x = layer(x)
            else:
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.deep_prompt_embeddings[i-1].expand(B, -1, -1)
                    ),
                    x[:, (1 + self.num_tokens):, :]
                ), dim=1)
                x = x.permute(1, 0, 2)  # NLD -> LND


                if self.prompt_config.FILTER:
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self._filter_ops(i, x)
                    x = x.permute(1, 0, 2)  # NLD -> LND

                x = layer(x)

        # before classifier
        if self.prompt_config.FILTER:
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self._filter_ops(-1, x)
            x = x.permute(1, 0, 2)  # NLD -> LND

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        return x

def build_model(model_type, clip_model, prompt_cfg):
    if "vitb" in model_type:
        return vit_base_patch16(clip_model, prompt_cfg)


def vit_base_patch16(clip_model, prompt_config, **kwargs):
    model = CLIPImageEncoder(clip_model, prompt_config, **kwargs)
    return model
