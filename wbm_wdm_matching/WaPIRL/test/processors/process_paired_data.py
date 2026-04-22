import imp
from turtle import color
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import torch
import cv2

def preprocess_data_to_taget_size(wbm, target_size=(10, 10), method='nearest'):
    """
    将 WBM 调整为 10*10 尺寸
    
    Args:
        wbm: 原始 WBM, shape: (H, W) 或 (H, W, C)
        target_size: 目标尺寸 (10, 10)
        method: 插值方法, 'nearest' 推荐用于离散值图像
    
    Returns:
        resized: shape: (10, 10) 或 (10, 10, C)
    """
    if wbm.shape[:2] == target_size:
        return wbm.copy()
    
    # 使用最近邻插值，保持离散值不变
    if wbm.ndim == 2:
        resized = cv2.resize(wbm, target_size, interpolation=cv2.INTER_NEAREST)
    else:
        resized = cv2.resize(wbm, target_size, interpolation=cv2.INTER_NEAREST)
    
    return resized


def transfer_to_wdm(dir_path, save_path, wbm_target_size=(20, 20), wdm_target_size=(96, 96)):
    os.makedirs(os.path.join(save_path, 'wdm'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'wbm'), exist_ok=True)
    
    for dirpath, dirnames, filenames in os.walk(dir_path):
        # 复制所有文件
        count = 0
        with tqdm(total=len(filenames), desc=f"正在处理{save_path}") as pbar:
            for filename in filenames:
                count += 1
                src_file = os.path.join(dirpath, filename)
                wdm_dst_file = os.path.join(save_path, 'wdm', f'{count}.png')
                wbm_dst_file = os.path.join(save_path, 'wbm', f'{count}.png')

                arr = np.array(Image.open(src_file))
                wbm_arr = preprocess_data_to_taget_size(arr, target_size=wbm_target_size)
                wdm_arr = preprocess_data_to_taget_size(arr, target_size=wdm_target_size)

                wdm_img = Image.fromarray(wdm_arr, mode='L')  # 灰度模式
                wbm_img = Image.fromarray(wbm_arr, mode='L')  # 灰度模式
                
                wdm_img.save(wdm_dst_file)
                wbm_img.save(wbm_dst_file)
                pbar.update(1)


import os
import shutil
import random
from tqdm import tqdm

def split_paired_data(save_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, move=False):
    """
    将 save_path/wdm 和 save_path/wbm 中的配对数据划分为训练/验证/测试集。
    
    Args:
        save_path: 包含 wdm 和 wbm 子文件夹的路径（即 transfer_to_wdm 中的 save_path）
        train_ratio, val_ratio, test_ratio: 划分比例，和应为 1
        move: True 表示移动文件（原文件夹内不再保留），False 表示复制文件
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为 1"
    
    wdm_dir = os.path.join(save_path, 'wdm')
    wbm_dir = os.path.join(save_path, 'wbm')
    
    # 获取所有配对文件的序号（假设 wdm 目录下有所有图片）
    wdm_files = sorted([f for f in os.listdir(wdm_dir) if f.endswith('.png')])
    # 根据序号生成配对标识
    all_indices = [f.replace('.png', '') for f in wdm_files]
    
    # 随机打乱配对标识
    random.shuffle(all_indices)
    n_total = len(all_indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # 测试集用剩余部分（避免因取整丢失数据）
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:n_train+n_val]
    test_indices = all_indices[n_train+n_val:]
    
    # 创建目标目录结构
    for split in ['train', 'val', 'test']:
        for sub in ['wdm', 'wbm']:
            os.makedirs(os.path.join(save_path, split, sub), exist_ok=True)
    
    def process_split(indices, split_name):
        for idx in tqdm(indices, desc=f"处理 {split_name} 集"):
            # wdm 文件
            src_wdm = os.path.join(wdm_dir, f"{idx}.png")
            dst_wdm = os.path.join(save_path, split_name, 'wdm', f"{idx}.png")
            # wbm 文件
            src_wbm = os.path.join(wbm_dir, f"{idx}.png")
            dst_wbm = os.path.join(save_path, split_name, 'wbm', f"{idx}.png")
            
            if move:
                shutil.move(src_wdm, dst_wdm)
                shutil.move(src_wbm, dst_wbm)
            else:
                shutil.copy2(src_wdm, dst_wdm)
                shutil.copy2(src_wbm, dst_wbm)
    
    process_split(train_indices, 'train')
    process_split(val_indices, 'val')
    process_split(test_indices, 'test')
    
    print(f"划分完成：训练 {len(train_indices)}，验证 {len(val_indices)}，测试 {len(test_indices)}")

if __name__ == '__main__':
    directory_path = '../../data/wm811k/labeled/test'
    save_path = '../data/wm811k/paired_data(20&96)'
    transfer_to_wdm(directory_path, save_path, wbm_target_size=(20, 20), wdm_target_size=(96, 96))
    split_paired_data(save_path)