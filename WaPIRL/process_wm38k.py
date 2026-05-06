# -*- coding: utf-8 -*-

import os
import time
import argparse
import glob

import numpy as np

from PIL import Image


class WM38kProcessor(object):
    CLASS_NAMES = ['center', 'donut', 'edge-loc', 'edge-ring', 'loc', 'random', 'scratch', 'near-full']

    def __init__(self, npz_file: str):
        start_time = time.time()
        self.data = np.load(npz_file)
        self.wafer_maps = self.data['arr_0']
        self.labels = self.data['arr_1']
        print(f'Successfully loaded WM38k data. {time.time() - start_time:.2f}s')
        print(f'Total samples: {len(self.wafer_maps)}')
        print(f'Wafer map shape: {self.wafer_maps[0].shape}')
        print(f'Label shape: {self.labels[0].shape}')

    @staticmethod
    def save_image(arr: np.ndarray, filepath: str, vmin: int = 0, vmax: int = 3):
        scaled_arr = (arr / vmax) * 255
        img = Image.fromarray(scaled_arr.astype(np.uint8))
        img.save(filepath, dpi=(500, 500))

    def get_label_string(self, label: np.ndarray):
        indices = np.where(label == 1)[0]
        if len(indices) == 0:
            return 'unknown'
        return '_'.join([self.CLASS_NAMES[i] for i in indices])

    def write_images(self, root: str):
        os.makedirs(root, exist_ok=True)
        for i in range(len(self.wafer_maps)):
            label_str = self.get_label_string(self.labels[i])
            pngfile = os.path.join(root, label_str, f'{i:06}.png')
            os.makedirs(os.path.dirname(pngfile), exist_ok=True)
            self.save_image(self.wafer_maps[i], pngfile)
            if i % 1000 == 0:
                print(f"Progress: {i}/{len(self.wafer_maps)}")

    def write_all_images(self, root: str = './data/wm38k/images/'):
        os.makedirs(root, exist_ok=True)
        stats = {name: 0 for name in self.CLASS_NAMES}
        stats['unknown'] = 0

        for i in range(len(self.wafer_maps)):
            label_str = self.get_label_string(self.labels[i])
            pngfile = os.path.join(root, label_str, f'{i:06}.png')
            os.makedirs(os.path.dirname(pngfile), exist_ok=True)
            self.save_image(self.wafer_maps[i], pngfile)
            if label_str in stats:
                stats[label_str] += 1
            else:
                stats[label_str] = 1
            if i % 5000 == 0:
                print(f"Progress: {i}/{len(self.wafer_maps)}")

        print("\nStatistics:")
        for name, count in stats.items():
            print(f"  {name}: {count}")


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser("Process WM38k data to individual image files.", add_help=True)
        parser.add_argument('--npz_file', type=str, default='./data/wm38k/Wafer_Map_Datasets.npz')
        parser.add_argument('--output_root', type=str, default='./data/wm38k/images')
        return parser.parse_args()

    def check_files_exist_in_directory(directory: str, file_ext: str = 'png', recursive: bool = True):
        files = glob.glob(os.path.join(directory, f"**/*.{file_ext}"), recursive=recursive)
        return len(files) > 0

    args = parse_args()

    if not check_files_exist_in_directory(args.output_root):
        processor = WM38kProcessor(args.npz_file)
        processor.write_all_images(root=args.output_root)
    else:
        print(f"Images exist in `{args.output_root}`. Skipping...")
