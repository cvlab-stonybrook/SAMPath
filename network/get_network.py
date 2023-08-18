from copy import deepcopy
from functools import partial
from segment_anything.modeling.common import LayerNorm2d

import torch
from torch import nn

class EncoderWrapper(nn.Module):
    def __init__(self, model, ft_dim, out_dim, neck=True, re_norm = False, mean=None, std=None):
        super().__init__()
        self.model = model
        if neck:
            self.neck = nn.Sequential(
                nn.Conv2d(
                    ft_dim,
                    out_dim,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_dim),
                nn.Conv2d(
                    out_dim,
                    out_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_dim),
            )
        else:
            self.neck = None

        self.re_norm = re_norm

        self.register_buffer("in_mean", torch.Tensor((0.485, 0.456, 0.406)).view(-1, 1, 1), False)
        self.register_buffer("in_std", torch.Tensor((0.229, 0.224, 0.225)).view(-1, 1, 1), False)
        self.register_buffer("mean", torch.Tensor(mean).view(-1, 1, 1), False)
        self.register_buffer("std", torch.Tensor(std).view(-1, 1, 1), False)

    def forward(self, x, no_grad=True):
        if self.re_norm:
            x = x * self.in_std + self.in_mean
            x = (x - self.mean) / self.std
        if no_grad:
            with torch.no_grad():
                x = self.model(x, dense=True)
        else:
            x = self.model(x, dense=True)
        # x should be B, 4096, dim
        if self.neck is not None:
            x = x.permute(0, 2, 1)
            x = x.reshape(x.shape[0], -1, 64, 64)
            x = self.neck(x)
        return x


def get_hipt(pretrained=None, neck=True):
    from .hipt.vision_transformer import vit_small
    from .hipt.hipt_prompt import load_ssl_weights

    model = vit_small(patch_size=16)
    model = load_ssl_weights(model, pretrained)

    model = EncoderWrapper(model, 384, 256, neck=neck,
                           re_norm=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    return model