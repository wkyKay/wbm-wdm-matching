# -*- coding: utf-8 -*-
"""
将 npz 中的 wafer maps 保存为黑/灰/白三色 PNG 预览图。

颜色约定：
  0 = 背景 = 黑色
  1 = wafer 有效区域无缺陷 = 灰色
  2 = 缺陷 = 白色

用法（从 MixedWa/ 目录运行）：
  python processors/show_npz_wafers.py \
    --npz_file ../../data/synthetic_wdm_mixed_wm811k_512/synthetic_wdm.npz \
    --output_dir ../../data/synthetic_wdm_mixed_wm811k_512/preview_all
"""

import argparse
import math
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Save wafer maps from npz as black/gray/white PNG images')
    parser.add_argument('--npz_file', type=str, required=True,
                        help='输入 npz 文件，默认读取 arr_0；若无 arr_0 则读取第一个 key')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='PNG 输出目录')
    parser.add_argument('--key', type=str, default=None,
                        help='可选：指定 npz key；默认优先 arr_0')
    parser.add_argument('--limit', type=int, default=None,
                        help='可选：最多保存多少张；默认保存全部')
    parser.add_argument('--start_index', type=int, default=0,
                        help='从第几张开始保存')
    parser.add_argument('--prefix', type=str, default='wafer',
                        help='输出文件名前缀')
    parser.add_argument('--make_contact_sheet', action='store_true',
                        help='额外生成一张 contact_sheet.png 总览图')
    parser.add_argument('--contact_cols', type=int, default=10,
                        help='contact sheet 每行列数')
    parser.add_argument('--contact_cell_size', type=int, default=128,
                        help='contact sheet 中每张 wafer 的显示尺寸')
    return parser.parse_args()


def load_npz_array(npz_file: str, key: str = None) -> np.ndarray:
    data = np.load(npz_file)
    if key is None:
        key = 'arr_0' if 'arr_0' in data else list(data.keys())[0]
    if key not in data:
        raise KeyError(f'Key {key!r} not found in {npz_file}. Available keys: {list(data.keys())}')

    arrays = np.asarray(data[key])
    if arrays.ndim == 2:
        arrays = arrays[None, ...]
    if arrays.ndim == 4 and arrays.shape[-1] == 1:
        arrays = arrays[..., 0]
    if arrays.ndim != 3:
        raise ValueError(f'Expected array shape (N,H,W), got {arrays.shape}')
    return arrays


def wafer_to_preview(arr: np.ndarray) -> np.ndarray:
    """把 wafer map 映射为 0/127/255 灰度图。"""
    arr = np.asarray(arr)
    preview = np.zeros(arr.shape, dtype=np.uint8)
    preview[arr == 1] = 127
    preview[arr == 2] = 255

    # 兼容 binary defect map：只有 0/1 时，将 1 视为缺陷白色。
    unique = set(np.unique(arr).tolist())
    if unique.issubset({0, 1}):
        preview[arr == 1] = 255
    return preview


def save_contact_sheet(images, output_path: str, cols: int, cell_size: int):
    if len(images) == 0:
        return
    cols = max(1, cols)
    rows = int(math.ceil(len(images) / cols))
    sheet = np.zeros((rows * cell_size, cols * cell_size), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        resized = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST)
        y0 = row * cell_size
        x0 = col * cell_size
        sheet[y0:y0 + cell_size, x0:x0 + cell_size] = resized

    cv2.imwrite(output_path, sheet)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    arrays = load_npz_array(args.npz_file, args.key)
    start = max(0, args.start_index)
    end = len(arrays) if args.limit is None else min(len(arrays), start + max(0, args.limit))
    if start >= len(arrays):
        raise ValueError(f'--start_index {start} is out of range for {len(arrays)} wafers')

    contact_images = []
    for idx in range(start, end):
        preview = wafer_to_preview(arrays[idx])
        out_path = os.path.join(args.output_dir, f'{args.prefix}_{idx:06}.png')
        cv2.imwrite(out_path, preview)
        if args.make_contact_sheet:
            contact_images.append(preview)

    if args.make_contact_sheet:
        save_contact_sheet(
            contact_images,
            os.path.join(args.output_dir, 'contact_sheet.png'),
            args.contact_cols,
            args.contact_cell_size,
        )

    print(f'Saved {end - start} preview images to: {args.output_dir}')


if __name__ == '__main__':
    main()
