import os

import cv2
import pandas as pd
import torch
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import vision_transformer as vits
import vision_transformer4k as vits4k


def eval_transforms(is_imagenet=False, patch_size=256):
    if is_imagenet:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return eval_t

def generate_mask(img_arr):
    sat = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)[:, :, 1]

    sat[sat <= 15] = 0
    sat[sat > 15] = 1
    return sat


def get_vit256(pretrained_weights, arch='vit_small'):
    r"""
    Builds ViT-256 Model.

    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.

    Returns:
    - model256 (torch.nn): Initialized model.
    """

    checkpoint_key = 'teacher'

    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    # for p in model256.parameters():
    #     p.requires_grad = False
    # model256.eval()

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    return model256


def get_vit4k(pretrained_weights, arch='vit4k_xs'):
    r"""
    Builds ViT-4K Model.

    Args:
    - pretrained_weights (str): Path to ViT-4K Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.

    Returns:
    - model256 (torch.nn): Initialized model.
    """

    checkpoint_key = 'teacher'
    model4k = vits4k.__dict__[arch](num_classes=0)
    # for p in model4k.parameters():
    #     p.requires_grad = False
    # model4k.eval()

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model4k.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    return model4k


class HIPT_4K(torch.nn.Module):
    """
    HIPT Model (ViT_4K-256) for encoding non-square images (with [256 x 256] patch tokens), with
    [256 x 256] patch tokens encoded via ViT_256-16 using [16 x 16] patch tokens.
    """

    def __init__(self, ck_dir, feature_4k=True):
        super().__init__()
        model256_path = os.path.join(ck_dir, 'vit256_small_dino.pth')
        model4k_path = os.path.join(ck_dir, 'vit4k_xs_dino.pth')
        self.model256 = get_vit256(pretrained_weights=model256_path)
        self.model4k = get_vit4k(pretrained_weights=model4k_path)
        # self.patch_filter_params = patch_filter_params

        self.feature_4k = feature_4k


    def forward(self, x, mask=None):
        """
        Forward pass of HIPT (given an image tensor x), outputting the [CLS] token from ViT_4K.
        1. x is center-cropped such that the W / H is divisible by the patch token size in ViT_4K (e.g. - 256 x 256).
        2. x then gets unfolded into a "batch" of [256 x 256] images.
        3. A pretrained ViT_256-16 model extracts the CLS token from each [256 x 256] image in the batch.
        4. These batch-of-features are then reshaped into a 2D feature grid (of width "w_256" and height "h_256".)
        5. This feature grid is then used as the input to ViT_4K-256, outputting [CLS]_4K.

        Args:
          - x (torch.Tensor): [1 x C x W' x H'] image tensor.

        Return:
          - features_cls4k (torch.Tensor): [1 x 192] cls token (d_4k = 192 by default).
        """
        # batch_256, w_256, h_256 = self.prepare_img_tensor(x)  # 1. [1 x 3 x W x H].
        batch_256 = x
        w_256, h_256 = 16, 16

        batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)  # 2. [1 x 3 x w_256 x h_256 x 256 x 256]
        batch_256 = rearrange(batch_256,
                              'b c p1 p2 w h -> (b p1 p2) c w h')  # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)

        if mask is not None:
            if len(mask.shape) < 4:
                mask = mask.unsqueeze(dim=1)
            mask_256 = mask.unfold(2, 256, 256).unfold(3, 256, 256)  # 2. [1 x 3 x w_256 x h_256 x 256 x 256]
            mask_256 = rearrange(mask_256,
                                  'b c p1 p2 w h -> (b p1 p2) c w h')  # 2. [B x 3 x 256 x 256], where B = (1*w_256*h_256)
            mask_sum = torch.sum(mask_256, dim=(1, 2, 3))

            batch_256 = batch_256[mask_sum > 0.1 * 256 * 256, ...]

        features_cls256 = []
        for mini_bs in range(0, batch_256.shape[0],
                             256):  # 3. B may be too large for ViT_256. We further take minibatches of 256.
            minibatch_256 = batch_256[mini_bs:mini_bs + 256]  # .to(self.device256, non_blocking=True)
            features_cls256.append(self.model256(
                minibatch_256).detach())  # 3. Extracting ViT_256 features from [256 x 3 x 256 x 256] image batches.

        features_cls256 = torch.vstack(features_cls256)  # 3. [B x 384], where 384 == dim of ViT-256 [ClS] token.

        if self.feature_4k:
            features_cls256 = features_cls256.reshape(w_256, h_256, 384).transpose(0, 1).transpose(0, 2).unsqueeze(dim=0)
            # features_cls256 = features_cls256.to(self.device4k, non_blocking=True)  # 4. [1 x 384 x w_256 x h_256]
            features_cls4k = self.model4k.forward(features_cls256)  # 5. [1 x 192], where 192 == dim of ViT_4K [ClS] token.
        else:
            features_cls4k = features_cls256
        return features_cls4k