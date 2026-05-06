'''处理测试数据集：
    1. 将WB-811k拆分成WBM，WDM两部分 1:1（仅测试）
    2. WBM处理成10*10的三值图 0（异常）， 1（正常），2（背景）
        WDM处理成96*96的三值图 0（异常）， 1（正常），2（背景）
'''

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import glob
import time
import argparse

import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.logging import get_tqdm_config


class WM811kProcessor(object):
    def __init__(self, wm811k_file: str):

        start_time = time.time()
        self.data = pd.read_pickle(wm811k_file)
        print(f'Successively loaded WM811k data. {time.time() - start_time:.2f}s')

        self.data['labelString'] = self.data['failureType'].apply(self.getLabelString)           # ..., '-'
        self.data['trainTestLabel'] = self.data['trianTestLabel'].apply(self.getTrainTestLabel)  # -1, 0, 1

        self.data['waferMapDim'] = self.data['waferMap'].apply(lambda x: x.shape)
        self.data['waferMapSize'] = self.data['waferMapDim'].apply(lambda x: x[0] * x[1])
        self.data['lotName'] = self.data['lotName'].apply(lambda x: x.replace('lot', ''))
        self.data['waferIndex'] = self.data['waferIndex'].astype(int)

    @staticmethod
    def save_image(arr: np.ndarray, filepath: str = 'image.png', vmin: int = 0, vmax: int = 2):
        scaled_arr = (arr / vmax) * 255
        img = Image.fromarray(scaled_arr.astype(np.uint8))
        img.save(filepath, dpi=(500, 500))

    @staticmethod
    def load_image(filepath: str = 'image.png'):
        return Image.open(filepath)

    def write_images(self, root: str, indices: list or tuple):
        """Write wafer images to .png files."""
        os.makedirs(root, exist_ok=True)
        with tqdm.tqdm(**get_tqdm_config(total=len(indices), leave=True, color='yellow')) as pbar:
            for i, row in self.data.loc[indices].iterrows():
                pngfile = os.path.join(root, row['labelString'], f'{i:06}.png')
                os.makedirs(os.path.dirname(pngfile), exist_ok=True)
                self.save_image(row['waferMap'], pngfile)
                pbar.set_description_str(f" {root} - {i:06} ")
                pbar.update(1)

    def write_unlabeled_images(self,
                               root: str = './data/wm811k/unlabeled/',
                               train_size: float = 0.8,
                               valid_size: float = 0.1):
        """Write wafer images without labels."""
        test_size = 1 - train_size - valid_size

        # Get train / validation / test indices
        unlabeled_indices = self.data.loc[self.data['trainTestLabel'] == -1].index
        train_indices, temp_indices = train_test_split(
            unlabeled_indices,
            train_size=train_size,
            shuffle=True,
            random_state=2015010720,
        )
        valid_indices, test_indices = train_test_split(
            temp_indices,
            train_size=valid_size / (valid_size + test_size),
            shuffle=True,
            random_state=2015010720,
        )

        self.write_images(os.path.join(root, 'train'), train_indices)
        self.write_images(os.path.join(root, 'valid'), valid_indices)
        self.write_images(os.path.join(root, 'test'), test_indices)

    def write_labeled_images(self,
                             root: str = './data/wm811k/labeled/',
                             train_size: float = 0.8,
                             valid_size: float = 0.1):
        """Write wafer images with labels."""
        test_size = 1 - train_size - valid_size

        labeled_indices = self.data.loc[self.data['trainTestLabel'] != -1].index
        temp_indices, test_indices = train_test_split(
            labeled_indices,
            test_size=test_size,
            stratify=self.data.loc[labeled_indices, 'labelString'],
            shuffle=True,
            random_state=2015010720,
        )
        train_indices, valid_indices = train_test_split(
            temp_indices,
            test_size=valid_size/(train_size + valid_size),
            stratify=self.data.loc[temp_indices, 'labelString'],
            random_state=2015010720,
        )

        self.write_images(os.path.join(root, 'train'), train_indices)
        self.write_images(os.path.join(root, 'valid'), valid_indices)
        self.write_images(os.path.join(root, 'test'), test_indices)

    def _split_train_valid_test(self, indices, train_size, valid_size, random_state, stratify=None):
        """Helper function to split indices into train/valid/test."""
        test_size = 1 - train_size - valid_size

        if stratify is not None:
            # First split
            temp_indices, test_indices = train_test_split(
                indices,
                test_size=test_size,
                stratify=stratify,
                shuffle=True,
                random_state=random_state,
            )
            # Second split
            train_indices, valid_indices = train_test_split(
                temp_indices,
                test_size=valid_size / (train_size + valid_size),
                stratify=stratify.loc[temp_indices] if hasattr(stratify, 'loc') else None,
                shuffle=True,
                random_state=random_state,
            )
        else:
            train_indices, temp_indices = train_test_split(
                indices,
                train_size=train_size,
                shuffle=True,
                random_state=random_state,
            )
            valid_indices, test_indices = train_test_split(
                temp_indices,
                train_size=valid_size / (valid_size + test_size),
                shuffle=True,
                random_state=random_state,
            )

        return train_indices, valid_indices, test_indices

    def write_wbm_wdm_images(self,
                              wbm_root: str = '../data/wm811k/wbm/',
                              wdm_root: str = '../data/wm811k/wdm/',
                              train_size: float = 0.8,
                              valid_size: float = 0.1,
                              random_state: int = 42):
        """
        Randomly split all data into two equal halves (WBM and WDM),
        each containing labeled/unlabeled splits with train/valid/test.

        Directory structure:
        wbm/
            labeled/
                train/
                valid/
                test/
            unlabeled/
                train/
                valid/
                test/
        wdm/
            labeled/
                train/
                valid/
                test/
            unlabeled/
                train/
                valid/
                test/

        Args:
            wbm_root: Root directory for WBM dataset
            wdm_root: Root directory for WDM dataset
            train_size: Proportion for training set (e.g., 0.8)
            valid_size: Proportion for validation set (e.g., 0.1)
            random_state: Random seed for reproducibility
        """
        # Get labeled and unlabeled indices
        labeled_indices = self.data.loc[self.data['trainTestLabel'] != -1].index.tolist()
        unlabeled_indices = self.data.loc[self.data['trainTestLabel'] == -1].index.tolist()

        print(f"Total samples: {len(labeled_indices) + len(unlabeled_indices)}")
        print(f"  - Labeled: {len(labeled_indices)}")
        print(f"  - Unlabeled: {len(unlabeled_indices)}")

        # Step 1: Split labeled data into WBM and WDM (50%/50%)
        labeled_wbm, labeled_wdm = train_test_split(
            labeled_indices,
            train_size=0.5,
            shuffle=True,
            random_state=random_state,
        )
        print(f"\nLabeled split:")
        print(f"  - WBM labeled: {len(labeled_wbm)}")
        print(f"  - WDM labeled: {len(labeled_wdm)}")

        # Step 2: Split unlabeled data into WBM and WDM (50%/50%)
        unlabeled_wbm, unlabeled_wdm = train_test_split(
            unlabeled_indices,
            train_size=0.5,
            shuffle=True,
            random_state=random_state,
        )
        print(f"\nUnlabeled split:")
        print(f"  - WBM unlabeled: {len(unlabeled_wbm)}")
        print(f"  - WDM unlabeled: {len(unlabeled_wdm)}")

        # ===== WBM Processing =====
        print("\n" + "="*50)
        print("Processing WBM dataset...")
        print("="*50)

        # WBM Labeled split
        print("\n[WBM Labeled] Splitting into train/valid/test...")
        wbm_labeled_train, wbm_labeled_valid, wbm_labeled_test = self._split_train_valid_test(
            labeled_wbm,
            train_size=train_size,
            valid_size=valid_size,
            random_state=random_state,
            stratify=self.data.loc[labeled_wbm, 'labelString'] if len(labeled_wbm) > 0 else None,
        )
        print(f"  WBM labeled - train: {len(wbm_labeled_train)}, valid: {len(wbm_labeled_valid)}, test: {len(wbm_labeled_test)}")

        self.write_images(os.path.join(wbm_root, 'labeled', 'train'), wbm_labeled_train)
        self.write_images(os.path.join(wbm_root, 'labeled', 'valid'), wbm_labeled_valid)
        self.write_images(os.path.join(wbm_root, 'labeled', 'test'), wbm_labeled_test)

        # WBM Unlabeled split
        print("\n[WBM Unlabeled] Splitting into train/valid/test...")
        wbm_unlabeled_train, wbm_unlabeled_valid, wbm_unlabeled_test = self._split_train_valid_test(
            unlabeled_wbm,
            train_size=train_size,
            valid_size=valid_size,
            random_state=random_state,
            stratify=None,
        )
        print(f"  WBM unlabeled - train: {len(wbm_unlabeled_train)}, valid: {len(wbm_unlabeled_valid)}, test: {len(wbm_unlabeled_test)}")

        self.write_images(os.path.join(wbm_root, 'unlabeled', 'train'), wbm_unlabeled_train)
        self.write_images(os.path.join(wbm_root, 'unlabeled', 'valid'), wbm_unlabeled_valid)
        self.write_images(os.path.join(wbm_root, 'unlabeled', 'test'), wbm_unlabeled_test)

        # ===== WDM Processing =====
        print("\n" + "="*50)
        print("Processing WDM dataset...")
        print("="*50)

        # WDM Labeled split
        print("\n[WDM Labeled] Splitting into train/valid/test...")
        wdm_labeled_train, wdm_labeled_valid, wdm_labeled_test = self._split_train_valid_test(
            labeled_wdm,
            train_size=train_size,
            valid_size=valid_size,
            random_state=random_state,
            stratify=self.data.loc[labeled_wdm, 'labelString'] if len(labeled_wdm) > 0 else None,
        )
        print(f"  WDM labeled - train: {len(wdm_labeled_train)}, valid: {len(wdm_labeled_valid)}, test: {len(wdm_labeled_test)}")

        self.write_images(os.path.join(wdm_root, 'labeled', 'train'), wdm_labeled_train)
        self.write_images(os.path.join(wdm_root, 'labeled', 'valid'), wdm_labeled_valid)
        self.write_images(os.path.join(wdm_root, 'labeled', 'test'), wdm_labeled_test)

        # WDM Unlabeled split
        print("\n[WDM Unlabeled] Splitting into train/valid/test...")
        wdm_unlabeled_train, wdm_unlabeled_valid, wdm_unlabeled_test = self._split_train_valid_test(
            unlabeled_wdm,
            train_size=train_size,
            valid_size=valid_size,
            random_state=random_state,
            stratify=None,
        )
        print(f"  WDM unlabeled - train: {len(wdm_unlabeled_train)}, valid: {len(wdm_unlabeled_valid)}, test: {len(wdm_unlabeled_test)}")

        self.write_images(os.path.join(wdm_root, 'unlabeled', 'train'), wdm_unlabeled_train)
        self.write_images(os.path.join(wdm_root, 'unlabeled', 'valid'), wdm_unlabeled_valid)
        self.write_images(os.path.join(wdm_root, 'unlabeled', 'test'), wdm_unlabeled_test)

        # ===== Summary =====
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"WBM dataset saved to: {wbm_root}")
        print(f"WDM dataset saved to: {wdm_root}")

        wbm_total = len(wbm_labeled_train) + len(wbm_labeled_valid) + len(wbm_labeled_test) + \
                    len(wbm_unlabeled_train) + len(wbm_unlabeled_valid) + len(wbm_unlabeled_test)
        wdm_total = len(wdm_labeled_train) + len(wdm_labeled_valid) + len(wdm_labeled_test) + \
                    len(wdm_unlabeled_train) + len(wdm_unlabeled_valid) + len(wdm_unlabeled_test)

        print(f"\nTotal samples per dataset:")
        print(f"  WBM: {wbm_total} (labeled: {len(labeled_wbm)}, unlabeled: {len(unlabeled_wbm)})")
        print(f"  WDM: {wdm_total} (labeled: {len(labeled_wdm)}, unlabeled: {len(unlabeled_wdm)})")

    @staticmethod
    def nearest_interpolate(arr, s=(40, 40)):
        assert isinstance(arr, np.ndarray) and len(arr.shape) == 2
        ptt = torch.from_numpy(arr).view(1, 1, *arr.shape).float()
        return F.interpolate(ptt, size=s, mode='nearest').squeeze().long().numpy()

    @staticmethod
    def getLabelString(x):
        if len(x) == 1:
            ls = x[0][0].strip().lower()  # Labeled (9 classes)
        else:
            ls = '-'
        return ls

    @staticmethod
    def getTrainTestLabel(x):
        d = {
            'unlabeled': -1,  # 638,507
            'training': 0,    # 118,595
            'test': 1,        #  54,355
        }
        if len(x) == 1:
            lb = x[0][0].strip().lower()
        else:
            lb = 'unlabeled'
        return d[lb]


if __name__ == '__main__':

    def parse_args():
        """Parse command line arguments."""

        parser = argparse.ArgumentParser("Process WM-811k data to individual image files.", add_help=True)
        parser.add_argument('--labeled_root', type=str, default='../../data/wm811k/labeled')
        parser.add_argument('--unlabeled_root', type=str, default='../../data/wm811k/unlabeled')
        parser.add_argument('--wbm_root', type=str, default='../data/wm811k/wbm')
        parser.add_argument('--wdm_root', type=str, default='../data/wm811k/wdm')
        parser.add_argument('--labeled_train_size', type=float, default=0.8)
        parser.add_argument('--labeled_valid_size', type=float, default=0.1)
        parser.add_argument('--unlabeled_train_size', type=float, default=0.8)
        parser.add_argument('--unlabeled_valid_size', type=float, default=0.1)
        parser.add_argument('--wbm_wdm_train_size', type=float, default=0.8)
        parser.add_argument('--wbm_wdm_valid_size', type=float, default=0.1)
        parser.add_argument('--random_state', type=int, default=42)

        return parser.parse_args()

    def check_files_exist_in_directory(directory: str, file_ext: str = 'png', recursive: bool = True):
        """Check existence of files of specific types are under a directory"""
        files = glob.glob(os.path.join(directory, f"**/*.{file_ext}"), recursive=recursive)
        return len(files) > 0  # True if files exist, else False.

    args = parse_args()
    processor = WM811kProcessor(wm811k_file='../../data/wm811k/LSWMD.pkl')

    if not check_files_exist_in_directory(args.labeled_root):
        processor.write_labeled_images(root='../../data/wm811k/labeled/', train_size=0.8, valid_size=0.1)
    else:
        print(f"Labeled images exist in `{args.labeled_root}`. Skipping...")

    if not check_files_exist_in_directory(args.unlabeled_root):
        processor.write_unlabeled_images(root='../../data/wm811k/unlabeled/', train_size=0.8, valid_size=0.1)
    else:
        print(f"Unlabeled images exist in `{args.unlabeled_root}`. Skipping...")

    # WBM and WDM split with labeled/unlabeled distinction
    if not check_files_exist_in_directory(args.wbm_root) or not check_files_exist_in_directory(args.wdm_root):
        processor.write_wbm_wdm_images(
            wbm_root=args.wbm_root,
            wdm_root=args.wdm_root,
            train_size=args.wbm_wdm_train_size,
            valid_size=args.wbm_wdm_valid_size,
            random_state=args.random_state,
        )
    else:
        print(f"WBM and WDM images exist. Skipping...")
