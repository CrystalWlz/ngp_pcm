import os
import random
import time
from multiprocessing import Process

root_dir = '/data2_12t/dataset/OpenXD-OmniObject3D-New/raw/blender_renders/'
num_folders = 5000
gpus_to_use = [0,1]  # 要使用的GPU编号列表
resume_training = True  # 是否从中断处继续训练
completed_folders_file = 'completed_folders.txt'  # 已完成文件夹列表文件

# 获取所有文件夹路径
folders = os.listdir(root_dir)
folders = [os.path.join(root_dir, folder) for folder in folders if os.path.isdir(os.path.join(root_dir, folder))]

if resume_training:
    # 检查已完成文件夹列表文件是否存在
    if os.path.exists(completed_folders_file):
        with open(completed_folders_file, 'r') as f:
            completed_folders = f.read().splitlines()
    else:
        completed_folders = []
else:
    completed_folders = []

# 检查是否已完成训练
while len(completed_folders) < num_folders:
    # 随机选择未完成的文件夹
    remaining_folders = list(set(folders) - set(completed_folders))

    # 获取已使用的exp_name编号列表
    # used_exp_names = [os.path.basename(folder) for folder in completed_folders]

    # 创建训练函数
    def train_on_gpu(gpu_id, selected_folder, exp_name):
        render_dir = os.path.join(selected_folder, 'render')
            
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py --num_epochs 10 --root_dir {render_dir} --exp_name {exp_name} --dataset_name nerf"
        # 执行训练命令
        print(f"Starting training on GPU {gpu_id}: Experiment {exp_name}")
        start_time = time.time()
        os.system(command)
        end_time = time.time()
        # 估计剩余时间
        elapsed_time = end_time - start_time
        remaining_folders.remove(selected_folder)
        remaining_folders_count = len(remaining_folders)
        remaining_time = elapsed_time * remaining_folders_count
        print(f"Training completed on GPU {gpu_id}: Experiment {exp_name}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Estimated remaining time: {remaining_time:.2f} seconds")

    # 创建并行进程进行训练
    processes = []
    for gpu_id in gpus_to_use:
        # 检查是否已完成训练
        if len(completed_folders) >= num_folders:
            print(f"All training completed on GPU {gpu_id}")
            break

        # 检查是否还有剩余文件夹
        if not remaining_folders:
            print(f"No remaining folders on GPU {gpu_id}")
            break

        # 随机选择训练集文件夹
        selected_folder = random.choice(remaining_folders)


        # 生成新的exp_name编号
        exp_name = len(completed_folders)

        # 创建并启动进程
        p = Process(target=train_on_gpu, args=(gpu_id, selected_folder, exp_name))
        p.start()
        processes.append(p)

        # 记录已完成的文件夹和exp_name
        completed_folders.append(selected_folder)

        # 保存已完成文件夹列表到文件
        with open(completed_folders_file, 'w') as f:
            f.write('\n'.join(completed_folders))

        # 从待使用文件夹列表中移除选中的文件夹
        remaining_folders.remove(selected_folder)

    # 等待所有进程完成
    for p in processes:
        p.join()
        
print("All training completed.")