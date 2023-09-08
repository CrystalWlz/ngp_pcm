import os
import random

# 定义文件夹路径
folder_path = "/data2_12t/dataset/OpenXD-OmniObject3D-New/results_img"

# 获取二级文件夹列表
subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

# 循环遍历二级文件夹
for subfolder in subfolders:
    subfolder_path = os.path.join(folder_path, subfolder)
    print(subfolder_path)
    files = [file for file in os.listdir(subfolder_path) if file.endswith('.png')]

    # 随机选择10个文件并删除剩下的
    random_files = random.sample(files, 10)
    for file in files:
        file_path = os.path.join(subfolder_path, file)
        if file in random_files:
            print(f"保留文件: {file_path}")
        else:
            os.remove(file_path)
            print(f"删除文件: {file_path}")