from box import Box

config = {
    # "num_devices": 2,
    "batch_size": 6,
    "num_workers": 4,
    "out_dir": "/data07/shared/jzhang/result/Concept/sam_trial/CRAG",
    "opt": {
        "num_epochs": 60,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2, #1e-2,
        "precision": 32, # "16-mixed"
        "steps":  [23 * 50, 23 * 55],
        "warmup_steps": 46,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/data07/shared/jzhang/result/weights/SAM/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
        "prompt_dim": 256,
        "prompt_decoder": False,
        "dense_prompt_decoder": False,

        "extra_encoder": 'hipt',
        "extra_type": "fusion",
        "extra_checkpoint": "/data07/shared/jzhang/result/weights/HIPT/vit256_small_dino.pth",
    },
    "loss": {
        "focal_cof": 0.125,
        "dice_cof": 0.875,
        "ce_cof": 0.,
        "iou_cof": 0.0,
    },
    "dataset": {
        "dataset_root": "/data07/shared/jzhang/data/segmentation/CRAG_org/merged",
        "dataset_csv_path": "/data07/shared/jzhang/data/segmentation/CRAG_org/merged/cv.csv",
        "data_ext": ".png",
        "val_fold_id": 0,
        "num_classes": 3,

        "ignored_classes": None,
        "ignored_classes_metric": 1, # if we do not count background, set to 1 (bg class)
        "image_hw": (1536, 1536), # default is 1024, 1024

        "feature_input": False, # or "True" for *.pt features
        "dataset_mean": (0.485, 0.456, 0.406),
        "dataset_std": (0.229, 0.224, 0.225),
    }
}

cfg = Box(config)