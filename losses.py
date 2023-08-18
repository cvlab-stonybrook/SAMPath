import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss

# To deal with unlabeled region for losses: make prediciton & target all zero
# To deal with unlabeled region for metrics: torchmetrics support ignore_index

class SAMLoss(nn.Module):

    def __init__(self, focal_cof: float = 20., dice_cof: float = 1., ce_cof: float = 0.,  iou_cof: float = 1.):
        super().__init__()
        self.focal_cof = focal_cof
        self.dice_cof = dice_cof
        self.ce_cof = ce_cof
        self.iou_cof = iou_cof

        self.dice_loss_fn = DiceLoss(include_background=False, to_onehot_y=False, sigmoid=False, softmax=False)
        self.focal_loss_fn = FocalLoss(include_background=False, to_onehot_y=False)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    @torch.no_grad()
    def to_one_hot_label(self, targets, num_classes):
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_one_hot = torch.movedim(targets_one_hot, -1, 1)
        return targets_one_hot

    def forward(self, inputs, targets, iou_pred, ignored_masks=None):
        # masks for ignored regions
        if ignored_masks is not None:
            inputs = inputs * (1. - ignored_masks.expand_as(inputs))
            targets = targets * (1 - ignored_masks.long().squeeze(1))

        targets_one_hot = self.to_one_hot_label(targets, num_classes=inputs.shape[1])

        inputs_softmax = F.softmax(inputs, dim=1)

        dice = self.dice_loss_fn(inputs_softmax, targets_one_hot)
        focal = self.focal_loss_fn(inputs, targets_one_hot)


        iou_true = calc_iou(inputs_softmax, targets_one_hot)
        iou = F.mse_loss(iou_pred[:, 1:], iou_true[:, 1:]) # ignore background
        # iou = 0.

        ce_loss = self.ce_loss_fn(inputs, targets)

        total_loss = self.focal_cof * focal + self.dice_cof * dice + self.ce_cof * ce_loss + self.iou_cof * iou

        return {
            "loss": total_loss,
            "focal": focal,
            "dice": dice,
            "ce": ce_loss,
            "iou": iou
        }


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    # both are B, N_cls, H, W
    # pred_mask = F.softmax(pred_mask, dim=1)
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(2, 3))
    union = torch.sum(pred_mask, dim=(2, 3)) + torch.sum(gt_mask, dim=(2, 3)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(2)
    return batch_iou

