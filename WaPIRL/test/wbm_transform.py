# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import albumentations as A

from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.transforms_interface import ImageOnlyTransform


class ToWBM(BasicTransform):
    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super(ToWBM, self).__init__(always_apply, p)

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img: np.ndarray, **kwargs):  # pylint: disable=unused-argument
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
    """
    掩码伯努利噪声增强（与WM811K保持一致）
    - 只作用于非零像素
    - 伯努利分布采样决定哪些像素被噪声
    - 噪声值在1和2之间互换
    适用于: WBM格式图像 {0, 1, 2}
    """
    def __init__(self, noise: float = 0.05, always_apply: bool = False, p: float = 1.0):
        super(MaskedBernoulliNoise, self).__init__(always_apply, p)
        self.noise = noise
        self.min_ = 0
        self.max_ = 1
        self.bernoulli = Bernoulli(probs=noise)

    def apply(self, x: torch.Tensor, **kwargs):  # pylint: disable=unused-argument
        assert x.ndim == 3
        m = self.bernoulli.sample(x.size()).to(x.device)
        m = m * x.gt(0).float()
        noise_value = 1 + torch.randint_like(x, self.min_, self.max_ + 1).to(x.device)  # 1 or 2
        return x * (1 - m) + noise_value * m

    def get_params(self):
        return {'noise': self.noise}

    def get_transform_init_args_names(self):
        return ["noise"]

class PixelNoise(ImageOnlyTransform):
    """
    噪声增强：随机翻转部分像素
    适用于二值图 {0, 1} 或三值图 {0, 127, 255}
    """
    def __init__(self, prob: float = 0.05, always_apply: bool = False, p: float = 1.0):
        super(PixelNoise, self).__init__(always_apply, p)
        self.prob = prob

    def apply(self, img: np.ndarray, **kwargs):
        noisy = img.copy()
        mask = np.random.rand(*img.shape) < self.prob
        if img.max() <= 1:
            noisy[mask] = 1 - noisy[mask]
        else:
            pass_fail_mask = (img > 0) & mask
            noisy[pass_fail_mask] = 1 - noisy[pass_fail_mask] if noisy.max() <= 1 else 255 - noisy[pass_fail_mask]
        return noisy

    def get_transform_init_args_names(self):
        return ["prob"]
        
class WBM10x10Transform(object):
    """Transformations for wafer bin maps (10x10)."""
    def __init__(self,
                 size: tuple = (10, 10),
                 mode: str = 'test',
                 **kwargs):

        if isinstance(size, int):
            size = (size, size)
        defaults = dict(size=size, mode=mode)
        defaults.update(kwargs)
        self.defaults = defaults

        if mode == 'rotate':
            transform = self.rotate_transform(**defaults)
        elif mode == 'flip':
            transform = self.flip_transform(**defaults)
        elif mode == 'noise':
            transform = self.noise_transform(**defaults)
        elif mode == 'cutout':
            transform = self.cutout_transform(**defaults)
        elif mode == 'test':
            transform = self.test_transform(**defaults)
        elif mode in ['rotate+noise', 'noise+rotate']:
            transform = self.rotate_noise_transform(**defaults)
        elif mode in ['flip+rotate', 'rotate+flip']:
            transform = self.flip_rotate_transform(**defaults)
        elif mode in ['flip+noise', 'noise+flip']:
            transform = self.flip_noise_transform(**defaults)
        elif mode in ['flip+cutout', 'cutout+flip']:
            transform = self.flip_cutout_transform(**defaults)
        elif mode in ['rotate+cutout', 'cutout+rotate']:
            transform = self.rotate_cutout_transform(**defaults)
        elif mode in ['noise+cutout', 'cutout+noise']:
            transform = self.noise_cutout_transform(**defaults)
        else:
            raise NotImplementedError

        self.transform = A.Compose(transform)

    def __call__(self, img):
        return self.transform(image=img)['image']

    def __repr__(self):
        repr_str = self.__class__.__name__
        for k, v in self.defaults.items():
            repr_str += f"\n{k}: {v}"
        return repr_str

    @staticmethod
    def rotate_transform(size: tuple, **kwargs) -> list:
        """
        Rotation-based augmentation, with `albumentations`.
        Expects a 3D numpy array of shape [H, W, C] as input.
        """
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=360, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ToWBM(),
        ]

        return transform

    
    @staticmethod
    def flip_transform(size: tuple, **kwargs) -> list:
        """
        Flip-based augmentation (horizontal and vertical).
        """
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def noise_transform(size: tuple, noise: float = 0.05, **kwargs) -> list:
        """
        噪声增强：随机翻转部分像素
        """
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
            MaskedBernoulliNoise(noise=noise),
        ]
        return transform

    @staticmethod
    def cutout_transform(size: tuple, num_holes: int = 1, cut_ratio: float = 0.1, **kwargs) -> list:
        """
        Cutout-based augmentation.
        """
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        cut_h = max(1, cut_h)
        cut_w = max(1, cut_w)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM()
        ]

        return transform

    @staticmethod
    def test_transform(size: tuple, **kwargs) -> list:
        """
        Test transformation (no augmentation).
        """
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def rotate_noise_transform(size: tuple, noise: float = 0.05, **kwargs) -> list:
        """
        Combined rotation and noise augmentation.
        """
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=360, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def flip_rotate_transform(size: tuple, **kwargs) -> list:
        """
        Combined flip and rotation augmentation.
        """
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=360, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def flip_noise_transform(size: tuple, noise: float = 0.05, **kwargs) -> list:
        """
        Combined flip and noise augmentation.
        """
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def flip_cutout_transform(size: tuple,
                              num_holes: int = 1, cut_ratio: float = 0.1,
                              **kwargs) -> list:
        """
        Combined flip and cutout augmentation.
        """
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        cut_h = max(1, cut_h)
        cut_w = max(1, cut_w)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def rotate_cutout_transform(size: tuple,
                                num_holes: int = 1, cut_ratio: float = 0.1,
                                **kwargs) -> list:
        """
        Combined rotation and cutout augmentation.
        """
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        cut_h = max(1, cut_h)
        cut_w = max(1, cut_w)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Rotate(limit=360, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM(),
        ]

        return transform

    @staticmethod
    def noise_cutout_transform(size: tuple,
                               num_holes: int = 1, cut_ratio: float = 0.1,
                               noise: float = 0.05,
                               **kwargs) -> list:
        """
        Combined noise and cutout augmentation.
        """
        cut_h = int(size[0] * cut_ratio)
        cut_w = int(size[1] * cut_ratio)
        cut_h = max(1, cut_h)
        cut_w = max(1, cut_w)
        transform = [
            A.Resize(*size, interpolation=cv2.INTER_NEAREST),
            A.Cutout(num_holes=num_holes, max_h_size=cut_h, max_w_size=cut_w, fill_value=0, p=kwargs.get('cutout_p', 0.5)),
            ToWBM(),
        ]

        return transform