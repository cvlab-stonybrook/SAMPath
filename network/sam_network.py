import copy
from functools import reduce
from operator import mul
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything.modeling.common import LayerNorm2d


class PromptSAM(nn.Module):

    def __init__(
            self,
            model_type: str = "vit_b",
            checkpoint: str = "",
            prompt_dim: int = 256,
            num_classes: int = 20,
            extra_encoder = None,
            freeze_image_encoder = True,
            freeze_prompt_encoder = True,
            freeze_mask_decoder = False,
            mask_HW = (1024, 1024),
            feature_input = False,
            prompt_decoder = False,
            dense_prompt_decoder=False,
            no_sam=False
    ):
        super().__init__()

        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.mask_HW = mask_HW
        self.feature_input = feature_input

        self.extra_encoder = extra_encoder
        # change prompt
        mask_tokens = nn.Embedding(num_classes + 1, prompt_dim)
        self.model.mask_decoder.mask_tokens = mask_tokens
        self.model.mask_decoder.num_mask_tokens = num_classes + 1

        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList(
            [
                # self.model.mask_decoder.output_hypernetworks_mlps[0].clone()
                copy.deepcopy(self.model.mask_decoder.output_hypernetworks_mlps[0])
                for i in range(self.model.mask_decoder.num_mask_tokens)
            ]
        )

        self.model.mask_decoder.iou_prediction_head.layers[-1] = nn.Linear(prompt_dim,
                                                                           self.model.mask_decoder.num_mask_tokens)

        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.dense_prompt_decoder = None
        if dense_prompt_decoder:
            decoder_layer = nn.TransformerDecoderLayer(d_model=prompt_dim, nhead=8)
            self.dense_prompt_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.no_sam = no_sam


    def forward(self, images):
        H, W = self.mask_HW

        if not self.feature_input:
            if images.shape[-2] != 1024 or images.shape[-1] != 1024:
                images = F.interpolate(images, (1024, 1024), mode="bilinear", align_corners=False)

            if not self.no_sam:
                with torch.no_grad():
                    image_embeddings = self.model.image_encoder(images)

        if self.extra_encoder is not None:
            extra_image_embeddings = self.extra_encoder(images)
            if self.no_sam:
                image_embeddings = extra_image_embeddings
                # print(image_embeddings.shape)
            else:
                image_embeddings = image_embeddings + extra_image_embeddings

        pred_masks = []
        ious = []
        for embedding in image_embeddings: #zip(image_embeddings):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            if self.dense_prompt_decoder is not None:
                # img embedding dim, H, W ->  HW, dim
                embedding_img = embedding.flatten(1).permute(1, 0)
                # sparse_embeddings: Ncls + 1, dim ->  Ncls, dim
                sparse_embeddings_v = self.model.mask_decoder.mask_tokens.weight.clone()
                # org dense_embeddings shape: 1, 256, 64, 64, now it is 4094, 256
                org_shape = dense_embeddings.shape
                dense_embeddings_gen = self.dense_prompt_decoder(embedding_img, sparse_embeddings_v)
                dense_embeddings_gen = dense_embeddings_gen.permute(1, 0).reshape(*org_shape)
                dense_embeddings = dense_embeddings + dense_embeddings_gen

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(0))
            ious.append(iou_predictions.reshape(-1, 1))

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)

class PromptSAMLateFusion(nn.Module):
    def __init__(
            self,
            model_type: str = "vit_b",
            checkpoint: str = "",
            prompt_dim: int = 256,
            num_classes: int = 20,
            extra_encoder = None,
            freeze_image_encoder = True,
            freeze_prompt_encoder = True,
            freeze_mask_decoder = False,
            mask_HW = (1024, 1024),
            feature_input = False,
            prompt_decoder = False,
            dense_prompt_decoder=False,
            no_sam=False,
    ):
        super().__init__()

        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.mask_HW = mask_HW
        self.feature_input = feature_input

        self.extra_encoder = extra_encoder
        # change prompt
        mask_tokens = nn.Embedding(num_classes + 1, prompt_dim)
        self.model.mask_decoder.mask_tokens = mask_tokens
        self.model.mask_decoder.num_mask_tokens = num_classes + 1

        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList(
            [
                # self.model.mask_decoder.output_hypernetworks_mlps[0].clone()
                copy.deepcopy(self.model.mask_decoder.output_hypernetworks_mlps[0])
                for i in range(self.model.mask_decoder.num_mask_tokens)
            ]
        )

        self.model.mask_decoder.iou_prediction_head.layers[-1] = nn.Linear(prompt_dim,
                                                                           self.model.mask_decoder.num_mask_tokens)

        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.fusion_neck = nn.Sequential(
            nn.Conv2d(
                768 + 384,
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

        self.dense_prompt_decoder = None
        if dense_prompt_decoder:
            decoder_layer = nn.TransformerDecoderLayer(d_model=prompt_dim, nhead=8)
            self.dense_prompt_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

    def forward(self, images):
        H, W = self.mask_HW

        if not self.feature_input:
            if images.shape[-2] != 1024 or images.shape[-1] != 1024:
                images = F.interpolate(images, (1024, 1024), mode="bilinear", align_corners=False)

            with torch.no_grad():
                image_embeddings = self.model.image_encoder(images, no_neck=True)

        if self.extra_encoder is not None:
            ex_embed = self.extra_encoder(images)
            ex_embed = ex_embed.reshape(ex_embed.shape[0], 64, 64, -1)
            image_embeddings = torch.cat((image_embeddings, ex_embed), dim=-1)
            image_embeddings = self.fusion_neck(image_embeddings.permute(0, 3, 1, 2))

        pred_masks = []
        ious = []
        for embedding in image_embeddings: #zip(image_embeddings):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            if self.dense_prompt_decoder is not None:
                # img embedding dim, H, W ->  HW, dim
                embedding_img = embedding.flatten(1).permute(1, 0)
                # sparse_embeddings: Ncls + 1, dim ->  Ncls, dim
                sparse_embeddings_v = self.model.mask_decoder.mask_tokens.weight.clone()
                #org dense_embeddings shape: 1, 256, 64, 64, now it is 4094, 256
                org_shape = dense_embeddings.shape
                dense_embeddings_gen = self.dense_prompt_decoder(embedding_img, sparse_embeddings_v)
                dense_embeddings_gen = dense_embeddings_gen.permute(1, 0).reshape(*org_shape)
                dense_embeddings = dense_embeddings + dense_embeddings_gen

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(0))
            ious.append(iou_predictions.reshape(-1, 1))

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)