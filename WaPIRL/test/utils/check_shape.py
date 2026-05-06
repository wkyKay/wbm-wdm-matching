import PIL.Image as Image
import numpy as np

src_file1 = "/data/wangkaiyan2024/wbm_wdm_matching/WaPIRL/test/data/wm811k/paired_data/wbm/1.png"
src_file2 = "/data/wangkaiyan2024/wbm_wdm_matching/WaPIRL/test/data/wm811k/paired_data/wdm/1.png"
arr1 = np.array(Image.open(src_file1))
print(arr1.shape)
arr2 = np.array(Image.open(src_file2))
print(arr2.shape)
