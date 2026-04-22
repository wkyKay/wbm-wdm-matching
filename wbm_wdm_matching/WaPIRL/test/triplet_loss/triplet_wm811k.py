# -*- coding: utf-8 -*-

import os
import glob
import pathlib

import numpy as np
import torch
import cv2

from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class PairedWBMWDDataset(Dataset):
    """
    配对的 WBM-WDM 数据集
    
    假设数据已经正确配对
    """
    
    def __init__(self, 
                 wbm_data,      # List[np.ndarray] 或 np.ndarray
                 wdm_data,      # List[np.ndarray] 或 np.ndarray
                 transform_wbm=None,
                 transform_wdm=None):
        """
        Args:
            wbm_data: WBM数据列表，长度N
            wdm_data: WDM数据列表，长度N，第i个与wbm_data[i]配对
            transform_wbm: WBM的数据增强
            transform_wdm: WDM的数据增强
        """
        assert len(wbm_data) == len(wdm_data)
        self.wbm_data = wbm_data
        self.wdm_data = wdm_data
        self.transform_wbm = transform_wbm
        self.transform_wdm = transform_wdm
    
    def __len__(self):
        return len(self.wbm_data)
    
    def __getitem__(self, idx):
        wbm = self.wbm_data[idx]
        wdm = self.wdm_data[idx]
        
        # 应用增强
        if self.transform_wbm:
            wbm = self.transform_wbm(wbm)
        if self.transform_wdm:
            wdm = self.transform_wdm(wdm)
        
        return {
            'wbm': torch.FloatTensor(wbm),
            'wdm': torch.FloatTensor(wdm),
            'idx': idx
        }


class WM811KForTriplet(Dataset):
    def __init__(self, wbm_root, wdm_root, proportion=1.0, wbm_transform=None, wdm_transform=None, decouple_input: bool = True,**kwargs):
        super(WM811KForTriplet, self).__init__()
        
        self.wbm_root = wbm_root
        self.wdm_root = wdm_root
        self.proportion = proportion
        self.wbm_transform = wbm_transform
        self.wdm_transform = wdm_transform
        self.decouple_input = decouple_input

        wbm_images  = sorted(glob.glob(os.path.join(wbm_root, '**/*.png'), recursive=True))
        wdm_images  = sorted(glob.glob(os.path.join(wdm_root, '**/*.png'), recursive=True))
        # assert len(wbm_images) == len(wdm_images)
        samples = list(zip(wbm_images, wdm_images))    

        if self.proportion < 1.0:
            # Randomly sample a proportion of the data
            self.samples, _ = train_test_split(
                samples,
                train_size=self.proportion,
                # stratify=[s[1] for s in samples],
                shuffle=True,
                random_state=1993 + kwargs.get('seed', 0),
            )
        else:
            self.samples = samples

    def __getitem__(self, idx):
        wbm_path, wdm_path = self.samples[idx]
        wbm_img = self.load_image_cv2(wbm_path)
        wdm_img = self.load_image_cv2(wdm_path)
        # filename = os.path.basename(wbm_path) 


        if self.wbm_transform is not None:
            wbm = self.wbm_transform(wbm_img)

        if self.wdm_transform is not None:
            wdm = self.wdm_transform(wdm_img)

        if self.decouple_input:
            wbm = self.decouple_mask(wbm)
            wdm = self.decouple_mask(wdm)

        return dict(wbm=wbm, wdm=wdm, idx=idx, wdm_path=wdm_path, wbm_path=wbm_path)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image_pil(filepath: str):
        """Load image with PIL. Use with `torchvision`."""
        return Image.open(filepath)

    @staticmethod
    def load_image_cv2(filepath: str):
        """Load image with cv2. Use with `albumentations`."""
        out = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 2D; (H, W)
        return np.expand_dims(out, axis=2)                # 3D; (H, W, 1)

    @staticmethod
    def decouple_mask(x: torch.Tensor):
        """
        Decouple input with existence mask.
        Defect bins = 2, Normal bins = 1, Null bins = 0
        """
        m = x.gt(0).float()
        x = torch.clamp(x - 1, min=0., max=1.)

        return torch.cat([x, m], dim=0)
