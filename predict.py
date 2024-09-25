import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from network.sam_network import PromptSAM, PromptSAMLateFusion
from pl_module_sam_seg import SamSeg
import albumentations
from torch.utils.data import Dataset, DataLoader


def get_augmentation(cfg):
    W, H = cfg.dataset.image_hw if cfg.dataset.image_hw is not None else (1024, 1024)
    transform_test_fn = albumentations.Compose([
        albumentations.Resize(H, W),
    ])
    return transform_test_fn


def get_model(cfg, pretrained=None):
    if cfg.model.extra_encoder is not None:
        print("Using %s as an extra encoder" % cfg.model.extra_encoder)
        neck = True if cfg.model.extra_type == 'plus' else False
        if cfg.model.extra_encoder == 'hipt':
            from network.get_network import get_hipt
            extra_encoder = get_hipt(cfg.model.extra_checkpoint, neck=neck)
        else:
            raise NotImplementedError
    else:
        extra_encoder = None
    if cfg.model.extra_type in ['plus']:
        MODEL = PromptSAM
    elif cfg.model.extra_type in ['fusion']:
        MODEL = PromptSAMLateFusion
    else:
        raise NotImplementedError

    model = MODEL(
        model_type = cfg.model.type,
        checkpoint = cfg.model.checkpoint,
        prompt_dim = cfg.model.prompt_dim,
        num_classes = cfg.dataset.num_classes,
        extra_encoder = extra_encoder,
        freeze_image_encoder = cfg.model.freeze.image_encoder,
        freeze_prompt_encoder = cfg.model.freeze.prompt_encoder,
        freeze_mask_decoder = cfg.model.freeze.mask_decoder,
        mask_HW = cfg.dataset.image_hw,
        feature_input = cfg.dataset.feature_input,
        prompt_decoder = cfg.model.prompt_decoder,
        dense_prompt_decoder=cfg.model.dense_prompt_decoder,
        no_sam=cfg.model.no_sam if "no_sam" in cfg.model else None
    )
    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
        state_dict = {k[len('model.'):]:v for k, v in state_dict.items() if k.startswith('model.')}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading weights from %s got msg: %s" % (pretrained, msg))
    return model

def get_data_module(cfg):
    from image_mask_dataset import GeneralDataModule, ImageMaskDataset, FtMaskDataset
    augs = get_augmentation(cfg)
    common_cfg_dic = {
        "dataset_root": cfg.dataset.dataset_root,
        "dataset_csv_path": cfg.dataset.dataset_csv_path,
        "val_fold_id": cfg.dataset.val_fold_id,
        "data_ext": ".jpg" if "data_ext" not in cfg.dataset else cfg.dataset.data_ext,
        "dataset_mean": cfg.dataset.dataset_mean,
        "dataset_std": cfg.dataset.dataset_std,
        "ignored_classes": cfg.dataset.ignored_classes,  # only supports None, 0 or [0, ...]
    }
    if cfg.dataset.feature_input is True:
        dataset_cls = FtMaskDataset
    else:
        dataset_cls = ImageMaskDataset

    data_module = GeneralDataModule(common_cfg_dic, dataset_cls, cus_transforms=augs,
                                    batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return data_module

def get_pl_module(cfg, model, metrics):
    pl_module = SamSeg(
        cfg = cfg,
        sam_model = model,
        metrics = metrics,
        num_classes = cfg.dataset.num_classes,
        focal_cof = cfg.loss.focal_cof,
        dice_cof = cfg.loss.dice_cof,
        ce_cof=cfg.loss.ce_cof,
        iou_cof = cfg.loss.iou_cof,
        lr = cfg.opt.learning_rate,
        weight_decay = cfg.opt.weight_decay,
        lr_steps =  cfg.opt.steps,
        warmup_steps=cfg.opt.warmup_steps,
        ignored_index=cfg.dataset.ignored_classes_metric,
    )
    return pl_module

def main(cfg, args):
    from image_mask_dataset import PredictionDataset
    dataset = PredictionDataset(args.input_dir, data_ext=args.data_ext, augmentation=get_augmentation(cfg),
                                dataset_mean=cfg.dataset.dataset_mean, dataset_std=cfg.dataset.dataset_std)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    sam_model = get_model(cfg, pretrained=args.pretrained)

    pl_module = get_pl_module(cfg, model=sam_model, metrics=None)

    # logger = WandbLogger(project=cfg.project, name=cfg.name, save_dir=cfg.out_dir, log_model=False)
    #
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')

    accumulate_grad_batches = cfg.accumulate_grad_batches if "accumulate_grad_batches" in cfg else 1

    trainer = Trainer(default_root_dir=os.path.join(args.output_dir, "log"),
                      devices=cfg.devices,
                      max_epochs=cfg.opt.num_epochs,
                      accelerator="gpu", #strategy="auto",
                      #strategy='ddp_find_unused_parameters_true',
                      log_every_n_steps=20, num_sanity_val_steps=0,
                      precision=cfg.opt.precision,
                      accumulate_grad_batches=accumulate_grad_batches,
                      fast_dev_run=False)

    pred_masks = trainer.predict(pl_module, dataloaders=dataloader)
    pred_masks = torch.cat(pred_masks, dim=0).cpu()
    print(pred_masks.shape)
    os.makedirs(args.output_dir, exist_ok=True)
    for f, pmask in zip(dataset.img_list, pred_masks):
        pmask = pmask.numpy().astype(np.uint8)
        out_f = os.path.join(args.output_dir, f[:-len(args.data_ext)] + "_mask.png")
        cv2.imwrite(out_f, pmask)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--data_ext", type=str, default=".jpg")

    parser.add_argument("--output_dir", default=None)
    parser.add_argument('--devices', type=lambda s: [int(item) for item in s.split(',')], default=[0])
    # parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    module = __import__(args.config, globals(), locals(), ['cfg'])
    cfg = module.cfg

    cfg["devices"] = args.devices
    # cfg["seed"] = args.seed

    # seed_everything(cfg["seed"])
    print(cfg)
    main(cfg, args)
    # print(cfg)
