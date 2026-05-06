import imp
from turtle import color
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import torch
import cv2

def preprocess_wbm_to_taget_size(wbm, target_size=(10, 10), method='nearest'):
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

def downsample_sliding(arr, kernel_size=2, stride=2, colors=[0, 127, 255], strategy='white_priority'):
    """
    使用滑动窗口对三值数组下采样。
    
    参数:
        arr: 输入数组 (h, w)
        kernel_size: 窗口大小（整数或元组）
        stride: 步长（整数或元组）
        colors: [背景值, 中间值, 前景值]，默认为 [0, 127, 255]
        strategy: 池化策略
            'white_priority' - 窗口内有白色(255)则输出白色，否则有0则输出0，否则灰色(127)
            'zero_dominant'   - 窗口内有0则输出0，否则取众数
            'max'             - 最大值池化
            'min'             - 最小值池化
            'mean'            - 平均值池化（会四舍五入到三值）
    返回:
        result: (out_h, out_w) 的数组
    """
    bg, gray, white = colors  # 0, 127, 255
    
    # 处理参数
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size
    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride
    
    h, w = arr.shape
    # 计算输出尺寸（向下取整，不填充）
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"窗口大小({kh},{kw})或步长({sh},{sw})过大，无法生成有效输出")
    
    result = np.zeros((out_h, out_w), dtype=arr.dtype)
    
    for i in range(out_h):
        for j in range(out_w):
            win = arr[i*sh : i*sh+kh, j*sw : j*sw+kw]
            
            if strategy == 'white_priority':
                # 白色优先：有白色则白色，否则有0则0，否则灰色
                if np.any(win == white):
                    result[i, j] = white
                elif np.any(win == bg):
                    result[i, j] = bg
                else:
                    result[i, j] = gray
                    
            elif strategy == 'zero_dominant':
                if np.any(win == bg):
                    result[i, j] = bg
                else:
                    cnt_gray = np.count_nonzero(win == gray)
                    cnt_white = np.count_nonzero(win == white)
                    result[i, j] = gray if cnt_gray >= cnt_white else white
                    
            elif strategy == 'max':
                result[i, j] = np.max(win)
            elif strategy == 'min':
                result[i, j] = np.min(win)
            elif strategy == 'mean':
                mean_val = np.mean(win)
                if mean_val < (gray + bg)/2:
                    result[i, j] = bg
                elif mean_val < (white + gray)/2:
                    result[i, j] = gray
                else:
                    result[i, j] = white
            else:
                raise ValueError(f"未知策略: {strategy}")
    return result


def adaptive_pool(arr, output_size, colors=[0, 127, 255], strategy='white_priority'):
    """
    自适应池化，将任意大小的三值数组下采样为 output_size * output_size。
    每个输出像素对应输入中的一个区域，区域大小由输入/输出比例自动确定。
    
    参数:
        arr: 输入数组 (h, w)
        output_size: 整数或元组 (out_h, out_w)，表示输出形状
        colors: [背景值, 中间值, 前景值]，默认为 [0, 127, 255]
        strategy: 池化策略，同 downsample_sliding 中的定义
            'white_priority' - 区域内有白色则白色，否则有0则0，否则灰色
            'zero_dominant'   - 区域内有0则0，否则取众数
            'max', 'min', 'mean' 等
    返回:
        result: (out_h, out_w) 的数组
    """
    bg, gray, white = colors
    h, w = arr.shape
    
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size
    
    # 计算每个输出像素对应的输入区域索引
    # 使用线性映射，使得输出像素 (i,j) 对应输入区域的 [start_h, end_h) 和 [start_w, end_w)
    # 采用向下取整的方式，保证覆盖所有输入像素，且区域大小差异不超过1
    h_step = h / out_h
    w_step = w / out_w
    
    result = np.zeros((out_h, out_w), dtype=arr.dtype)
    
    for i in range(out_h):
        start_h = int(i * h_step)
        end_h = int((i + 1) * h_step) if i < out_h - 1 else h
        for j in range(out_w):
            start_w = int(j * w_step)
            end_w = int((j + 1) * w_step) if j < out_w - 1 else w
            win = arr[start_h:end_h, start_w:end_w]
            
            if strategy == 'white_priority':
                if np.any(win == white):
                    result[i, j] = white
                elif np.any(win == bg):
                    result[i, j] = bg
                else:
                    result[i, j] = gray
                    
            elif strategy == 'zero_dominant':
                if np.any(win == bg):
                    result[i, j] = bg
                else:
                    cnt_gray = np.count_nonzero(win == gray)
                    cnt_white = np.count_nonzero(win == white)
                    result[i, j] = gray if cnt_gray >= cnt_white else white
                    
            elif strategy == 'max':
                result[i, j] = np.max(win)
            elif strategy == 'min':
                result[i, j] = np.min(win)
            elif strategy == 'mean':
                mean_val = np.mean(win)
                if mean_val < (gray + bg)/2:
                    result[i, j] = bg
                elif mean_val < (white + gray)/2:
                    result[i, j] = gray
                else:
                    result[i, j] = white
            else:
                raise ValueError(f"未知策略: {strategy}")
    return result

# def transfer_to_wdm(file_path, save_path):
#     img = Image.open(file_path)  
#     # 转换为 numpy 数组
#     arr = np.array(img)
#     # wdm_arr = downsample_sliding(arr, kernel_size=3, stride=3, strategy="mean")
#     wdm_arr = preprocess_wbm_to_10x10(arr)
#     img = Image.fromarray(wdm_arr, mode='L')  # 灰度模式
#     img.save(save_path)


def transfer_to_wdm(dir_path, save_path, target_size=(40, 40)):
    os.makedirs(save_path, exist_ok=True)
    
    for dirpath, dirnames, filenames in os.walk(dir_path):
        # 计算当前目录相对于源根目录的路径
        rel_path = os.path.relpath(dirpath, directory_path)
        if rel_path == '.':
            # 当前是根目录本身，目标路径就是 save_path
            target_dir = save_path
        else:
            target_dir = os.path.join(save_path, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        # 复制所有文件
        with tqdm(total=len(filenames), desc=f"正在处理{target_dir}") as pbar:
            for filename in filenames:
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(target_dir, filename)
                arr = np.array(Image.open(src_file))
                wdm_arr = preprocess_wbm_to_taget_size(arr, target_size=target_size)
                img = Image.fromarray(wdm_arr, mode='L')  # 灰度模式
                img.save(dst_file)
                pbar.update(1)


if __name__ == '__main__':
    directory_path = '../data/wm811k/wbm'
    save_path = '../data/wm811k/wbm(20*20)'
    transfer_to_wdm(directory_path, save_path, target_size=(20, 20))