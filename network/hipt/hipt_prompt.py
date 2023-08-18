import os
import math
from functools import partial, reduce
from operator import mul

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn.modules.utils import _pair

from . import vision_transformer as vits
from . import vision_transformer4k as vits4k


class PromptedTransformer(vits.VisionTransformer):
    def __init__(
            self,
            vit_config,
            num_tokens=1,
            drop_out=0.,
            project_prompt_dim=-1,
            deep_prompt=False,
    ):
        super().__init__(**vit_config)
        self.vit_config = vit_config

        self.num_prefix_tokens = 1

        patch_size = _pair(vit_config["patch_size"])

        self.num_prompt_tokens = num_tokens  # number of prompted tokens
        self.deep_prompt = deep_prompt

        self.prompt_dropout = nn.Dropout(drop_out)

        # if project the prompt embeddings
        if project_prompt_dim > 0:
            # only for prepend / add
            prompt_dim = project_prompt_dim
            self.prompt_proj = nn.Linear(
                prompt_dim, vit_config["embed_dim"])
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = vit_config["embed_dim"]
            self.prompt_proj = nn.Identity()

        if num_tokens > 0:
            # initiate prompt:
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            pass

        if self.deep_prompt:  # noqa
            total_d_layer = vit_config["depth"] - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                total_d_layer, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.prepare_tokens(x)

        prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
        x = torch.cat((
            x[:, :self.num_prefix_tokens, :],
            prompt,
            x[:, self.num_prefix_tokens:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        # x = self.norm_pre(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            if self.num_prompt_tokens > 0:
                super().train(False)
                self.prompt_proj.train()
                self.prompt_dropout.train()
            else:
                super().train(mode)
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        # attn_weights = []
        hidden_states = None
        # weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config["depth"]

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.blocks[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :self.num_prefix_tokens, :],
                        deep_prompt_emb,
                        hidden_states[:, (self.num_prefix_tokens + self.num_prompt_tokens):, :]
                    ), dim=1)

                hidden_states = self.blocks[i](hidden_states)
        return hidden_states

    def forward_features(self, x):
        if self.num_prompt_tokens > 0:
            x = self.incorporate_prompt(x)
        else:
            x = self.prepare_tokens(x)

        if self.num_prompt_tokens > 0 and self.deep_prompt:
            x = self.forward_deep_prompt(x)
        else:
            for blk in self.blocks:
                x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        return x[:, 0]


def vit_small(num_tokens=1, drop_out=0., project_prompt_dim=-1, deep_prompt=False):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = PromptedTransformer(model_kwargs, num_tokens, drop_out, project_prompt_dim, deep_prompt)
    return model

class PromptedTransformer4k(vits4k.VisionTransformer4K):
    def __init__(
            self,
            vit_config,
            num_tokens=1,
            drop_out=0.,
            project_prompt_dim=-1,
            deep_prompt=False,
    ):
        super().__init__(**vit_config)
        self.vit_config = vit_config

        self.num_prefix_tokens = 1

        patch_size = _pair(vit_config["patch_size"])

        self.num_prompt_tokens = num_tokens  # number of prompted tokens
        self.deep_prompt = deep_prompt

        self.prompt_dropout = nn.Dropout(drop_out)

        # if project the prompt embeddings
        if project_prompt_dim > 0:
            # only for prepend / add
            prompt_dim = project_prompt_dim
            self.prompt_proj = nn.Linear(
                prompt_dim, vit_config["output_embed_dim"])
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = vit_config["output_embed_dim"]
            self.prompt_proj = nn.Identity()

        if num_tokens > 0:
            # initiate prompt:
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            pass

        if self.deep_prompt:  # noqa
            total_d_layer = vit_config["depth"] - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                total_d_layer, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.prepare_tokens(x)

        prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
        x = torch.cat((
            x[:, :self.num_prefix_tokens, :],
            prompt,
            x[:, self.num_prefix_tokens:, :]
        ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        # x = self.norm_pre(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            if self.num_prompt_tokens > 0:
                super().train(False)
                self.prompt_proj.train()
                self.prompt_dropout.train()
            else:
                super().train(mode)
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        # attn_weights = []
        hidden_states = None
        # weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config["depth"]

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.blocks[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :self.num_prefix_tokens, :],
                        deep_prompt_emb,
                        hidden_states[:, (self.num_prefix_tokens + self.num_prompt_tokens):, :]
                    ), dim=1)

                hidden_states = self.blocks[i](hidden_states)
        return hidden_states

    def forward_features(self, x):
        if self.num_prompt_tokens > 0:
            x = self.incorporate_prompt(x)
        else:
            x = self.prepare_tokens(x)

        if self.num_prompt_tokens > 0 and self.deep_prompt:
            x = self.forward_deep_prompt(x)
        else:
            for blk in self.blocks:
                x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        return x[:, 0]


def vit4k_xs(num_tokens=1, drop_out=0., project_prompt_dim=-1, deep_prompt=False):
    model_kwargs = dict(patch_size=16, input_embed_dim=384, output_embed_dim=192, depth=6, num_heads=6, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),)
    model = PromptedTransformer4k(model_kwargs, num_tokens, drop_out, project_prompt_dim, deep_prompt)
    return model


def load_ssl_weights(model, pretrained_weights):
    checkpoint_key = 'teacher'

    # model = vits.__dict__[arch](patch_size=16, num_classes=0)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    return model


class PromptHIPT4K(torch.nn.Module):
    """
    HIPT Model (ViT_4K-256) for encoding non-square images (with [256 x 256] patch tokens), with
    [256 x 256] patch tokens encoded via ViT_256-16 using [16 x 16] patch tokens.
    """

    def __init__(self, num_tokens=1, drop_out=0., project_prompt_dim=-1, deep_prompt=False, feature_4k=True,
                 w_256=16, h_256=16, patch_size=256):
        super().__init__()

        self.model256 = vit_small(num_tokens, drop_out, project_prompt_dim, deep_prompt)
        self.model4k = vit4k_xs(num_tokens, drop_out, project_prompt_dim, deep_prompt)
        # self.patch_filter_params = patch_filter_params
        self.num_features = self.model4k.num_features
        self.feature_4k = feature_4k

        self.w_256 = w_256
        self.h_256 = h_256
        self.patch_size = patch_size

    def load_weights(self, ck_dir):
        model256_path = os.path.join(ck_dir, 'vit256_small_dino.pth')
        model4k_path = os.path.join(ck_dir, 'vit4k_xs_dino.pth')
        self.model256 = load_ssl_weights(self.model256, model256_path)
        self.model4k = load_ssl_weights(self.model4k, model4k_path)

    def prepare_256(self, x):
        # 1. x= [B, 3, h, w]
        B, nc, h, w = x.shape
        x = x.reshape(B, nc, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        x = rearrange(x, 'b c p1 h p2 w -> (b p1 p2) c h w')
        return x

    def forward_256(self, x):
        x = self.model256(x)
        return x

    def prepare_4k(self, x):
        x = x.reshape(-1, self.h_256, self.w_256, x.shape[-1])
        return x

    def unprepare_4k(self, x):
        x = x.reshape(-1, x.shape[-1])
        return x

    def forward_4k(self, x):
        features_cls4k = self.model4k.forward(x)  # 5. [B x 192], where 192 == dim of ViT_4K [ClS] token.
        return features_cls4k


    def forward(self, x, mask=None, batch_size=64):
        x = self.prepare_256(x)
        x = torch.chunk(x, int(np.ceil(x.shape[0] / batch_size)), dim=0)
        x = torch.cat([self.forward_256(xx) for xx in x], dim=0)

        x = self.prepare_4k(x)
        x = torch.chunk(x, int(np.ceil(x.shape[0] / batch_size)), dim=0)
        x = torch.cat([self.forward_4k(xx) for xx in x], dim=0)
        return x


if __name__ == '__main__':
    # torch.cuda.set_device(2)
    model = PromptHIPT4K()
    model.load_weights("/home/jzhang/Projects/transformer/HIPT/HIPT_4K/Checkpoints")

    model = model.cuda()

    img = torch.rand(10, 3, 4096, 4096).cuda()

    with torch.no_grad():
        out = model(img)

    print(out.shape)