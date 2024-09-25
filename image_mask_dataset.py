import os

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import jpeg4py
from torchvision.transforms import transforms

SUB_FOLDER_IMAGE = "img"
SUB_FOLDER_MASK = "mask"


def read_rgb_img(p):
    if p.lower().endswith((".jpg", ".jpeg")):
        try:
            return jpeg4py.JPEG(p).decode()
        except:
            # cv2.setNumThreads(0)
            return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    else:
        # cv2.setNumThreads(0)
        return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def read_mask(p):
    return cv2.imread(p, cv2.IMREAD_UNCHANGED) #.astype(np.int_)


class ImageMaskDataset(Dataset):

    def __init__(
            self,
            dataset_root: str,
            dataset_csv_path: str,
            data_type: str,
            val_fold_id: int,
            augmentation=None,
            data_ext: str =".jpg",
            dataset_mean=(0.485, 0.456, 0.406),
            dataset_std=(0.229, 0.224, 0.225),
            ignored_classes=None, # only supports None, 0 or [0, ...]
    ):
        super().__init__()

        self.dataset_root = dataset_root
        self.dataset_csv_path = dataset_csv_path
        self.data_ext = data_ext
        self.augmentation = augmentation

        self.setup(data_type, val_fold_id)

        self.tensor_transforms = albumentations.Compose([
            albumentations.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),

        ])
        self.ignored_classes = ignored_classes

    def __len__(self):
        return len(self.img_list)

    def setup(self, data_type, val_fold_id):
        if data_type not in ['train', 'val', 'test']:
            raise Exception("Not supported dataset type. It should be train, val or test")
        self.data_type = data_type
        self.val_fold_id = val_fold_id
        if data_type == 'test':
            self.val_fold_id = -1

        if val_fold_id >= 0:
            self.img_list = self.read_cv_dataset_csv()
        else:
            if data_type == 'val':
                data_type = 'test'
                self.data_type = data_type
            self.img_list = self.read_dataset_csv()

    def read_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['test']:
            df = df[df['is_test'] > 0]
        else:  # train
            df = df[df['is_test'] == 0]
        return df

    def read_cv_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['val']:
            df = df[df['fold'] == self.val_fold_id]
        elif self.data_type in ['test']:
            df = df[df['fold'] < 0]
        else:
            df = df[df['fold'] > 0]
            df = df[df['fold'] != self.val_fold_id]
        return df

    def process_ignored_classes(self, mask):
        if self.ignored_classes is not None:
            if not isinstance(self.ignored_classes, (list, tuple)):
                self.ignored_classes = [self.ignored_classes]
            for cls in self.ignored_classes:
                if cls != 0:
                    mask[mask == cls] = 0
        else:
            mask += 1
        return mask

    def __getitem__(self, i):
        row = self.img_list.iloc[i]
        img_id = row['img_id']

        image = read_rgb_img(os.path.join(self.dataset_root, SUB_FOLDER_IMAGE, img_id + self.data_ext))
        mask = read_mask(os.path.join(self.dataset_root, SUB_FOLDER_MASK, img_id + ".png"))

        if self.augmentation is not None:
            ret = self.augmentation(image=image, mask=mask)
            image, mask = ret["image"], ret["mask"]

        mask = self.process_ignored_classes(mask)

        ret = self.tensor_transforms(image=image, mask=mask)
        image, mask = ret["image"], ret["mask"]

        return image, mask.long()

class FtMaskDataset(ImageMaskDataset):
    def __init__(
            self,
            dataset_root: str,
            dataset_csv_path: str,
            data_type: str,
            val_fold_id: int,
            augmentation = None,
            data_ext: str = ".pt", # only changed this
            dataset_mean = (0.485, 0.456, 0.406),
            dataset_std = (0.229, 0.224, 0.225),
            ignored_classes = None, # only supports None, 0 or [0, ...]
    ):
        super().__init__(
            dataset_root,
            dataset_csv_path,
            data_type,
            val_fold_id,
            augmentation,
            data_ext,
            dataset_mean,
            dataset_std,
            ignored_classes,
        )

    def __getitem__(self, i):
        row = self.img_list.iloc[i]
        img_id = row['img_id']

        image = torch.load(os.path.join(self.dataset_root, SUB_FOLDER_IMAGE, img_id + self.data_ext),
                           map_location='cpu')
        mask = read_mask(os.path.join(self.dataset_root, SUB_FOLDER_MASK, img_id + ".png"))

        mask = self.process_ignored_classes(mask)

        mask = torch.from_numpy(mask).long()

        return image, mask



class GeneralDataModule(LightningDataModule):
    def __init__(self, common_cfg_dic, dataset_classs, cus_transforms, batch_size, num_workers):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_train, self.dataset_val, self.dataset_test = self.initialize_dataset(common_cfg_dic,
                                                                                          dataset_classs,
                                                                                          cus_transforms)

    def initialize_dataset(self, common_cfg, DatasetCLS, cus_transforms):
        if cus_transforms is None:
            transforms_train, transforms_eval = None, None
        elif isinstance(cus_transforms, (list, tuple)):
            transforms_train = cus_transforms[0]
            transforms_eval = cus_transforms[1]
        else:
            transforms_train, transforms_eval = cus_transforms, cus_transforms

        dataset_train = DatasetCLS(**common_cfg, data_type="train", augmentation=transforms_train)
        dataset_val = DatasetCLS(**common_cfg, data_type="val", augmentation=transforms_eval)
        dataset_test = DatasetCLS(**common_cfg, data_type="test", augmentation=transforms_eval)
        return dataset_train, dataset_val, dataset_test

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers), \
            DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=self.num_workers)

class PredictionDataset(Dataset):

    def __init__(
            self,
            dataset_root: str,
            data_ext: str = ".jpg",
            augmentation=None,
            dataset_mean=(0.485, 0.456, 0.406),
            dataset_std=(0.229, 0.224, 0.225),
    ):
        super().__init__()

        self.dataset_root = dataset_root
        self.data_ext = data_ext
        self.augmentation = augmentation

        self.tensor_transforms = albumentations.Compose([
            albumentations.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),
        ])

        self.img_list = [f for f in os.listdir(self.dataset_root) if f.lower().endswith(data_ext)]

    def __len__(self):
        return len(self.img_list)

    def process_ignored_classes(self, mask):
        if self.ignored_classes is not None:
            if not isinstance(self.ignored_classes, (list, tuple)):
                self.ignored_classes = [self.ignored_classes]
            for cls in self.ignored_classes:
                if cls != 0:
                    mask[mask == cls] = 0
        else:
            mask += 1
        return mask

    def __getitem__(self, i):
        img_id = self.img_list[i]

        image = read_rgb_img(os.path.join(self.dataset_root, img_id))

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]


        image = self.tensor_transforms(image=image)["image"]
        return image