import os

def count_files_in_dir(path):
    count = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                count += 1
    return count

# 示例
folder_path = '/data/wangkaiyan2024/data/wm38k/images/unknown'  # 当前目录
print(count_files_in_dir(folder_path))