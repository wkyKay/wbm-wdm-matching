# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import albumentations as A

from torch.distributions import Bernoulli
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.transforms_interface import ImageOnlyTransform


class ToWBM(BasicTransform):
    """将 numpy 数组转换为 WBM 格式的 torch tensor，值域 {0, 1, 2}。"""
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super(ToWBM, self).__init__(always_apply, p)

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img: np.ndarray, **kwargs):
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, None]
            img = torch.from_numpy(img.transpose(2, 0, 1))
            if isinstance(img, torch.ByteTensor):
                img = img.float().div(255)
        return torch.ceil(img * 2)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}


class MaskedBernoulliNoise(ImageOnlyTransform):
    """仅对非零区域添加伯努利噪声。"""
    def __init__(self, noise: float, always_apply: bool = False, p: float = 1.0):
        super(MaskedBernoulliNoise, self).__init__(always_apply, p)
        self.noise = noise
        self.bernoulli = Bernoulli(probs=noise)

    def apply(self, x: torch.Tensor, **kwargs):
        assert x.ndim == 3
        m = self.bernoulli.sample(x.size()).to(x.device)
        m = m * x.gt(0).float()
        noise_value = 1 + torch.randint_like(x, 0, 2).to(x.device)
        return x * (1 - m) + noise_value * m

    def get_params(self):
        return {'noise': self.noise}


class WaferTransform(object):
    """
    晶圆图增强变换，支持以下模式：
      test, crop, shift, noise, rotate, cutout
      以及组合模式：crop+shift（位置感知训练专用）
    """
    def __init__(self, size: tuple = (96, 96), mode: str = 'test', **kwargs):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.mode = mode

        if mode == 'test':
            ops = self._test(size)
        elif mode == 'crop':
            ops = self._crop(size, **kwargs)
        elif mode == 'shift':
            ops = self._shift(size, **kwargs)
        elif mode == 'noise':
            ops = self._noise(size, **kwargs)
        elif mode == 'rotate':
            ops = self._rotate(size)
        elif mode == 'cutout':
            ops = self._cutout(size, **kwargs)
        elif mode in ('crop+shift', 'shift+crop'):
            ops = self._crop_shift(size, **kwargs)
        elif mode in ('crop+noise', 'noise+crop'):
            ops = self._crop_noise(size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown transform mode: {mode}")

        self.transform = A.Compose(ops)

    def __call__(self, img: np.ndarray):
        return self.transform(image=img)['image']

    @staticmethod
    def _test(size):
        return [A.Resize(*size, interpolation=cv2.INTER_NEAREST), ToWBM()]

    @staticmethod
    def _crop(size, scale=(0.5, 1.0), ratio=(0.9, 1.1), **kwargs):
        return [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio,
                                interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
        ]

    @staticmethod
    def _shift(size, shift: float = 0.25, **kwargs):
        return [
            A.ShiftScaleRotate(
                shift_limit=shift, scale_limit=0, rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0,
            ),
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

    @staticmethod
    def _noise(size, noise: float = 0.05, **kwargs):
        return [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]

    @staticmethod
    def _rotate(size):
        return [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=180, interpolation=cv2.INTER_NEAREST,
                     border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ToWBM(),
        ]

    @staticmethod
    def _cutout(size, num_holes: int = 4, cut_ratio: float = 0.2, **kwargs):
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        return [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h,
                     max_w_size=cut_w, fill_value=0, p=0.5),
            ToWBM(),
        ]

    @staticmethod
    def _crop_shift(size, scale=(0.5, 1.0), ratio=(0.9, 1.1), shift: float = 0.25, **kwargs):
        """crop + 大幅平移，用于阶段二位置感知训练的负样本生成。"""
        return [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio,
                                interpolation=cv2.INTER_NEAREST, p=1.0),
            A.ShiftScaleRotate(
                shift_limit=shift, scale_limit=0, rotate_limit=0,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0,
            ),
            ToWBM(),
        ]

    @staticmethod
    def _crop_noise(size, scale=(0.5, 1.0), ratio=(0.9, 1.1), noise: float = 0.05, **kwargs):
        return [
            A.RandomResizedCrop(*size, scale=scale, ratio=ratio,
                                interpolation=cv2.INTER_NEAREST, p=1.0),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]
