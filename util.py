import random
import torch
import numpy as np
import time
from loguru import logger
import shutil
import sys

def set_seed(seed=42):
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-gpu usage
    
    # Settings for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Random seed for the data-loader
    torch.Generator().manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def init_logger(fname=''):
    dir_name = './logs/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    time_stamp = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    logger.add('./logs/%s-%s-%s.log' % (time_stamp, os.path.basename(sys.argv[0])[:-3],fname),
               format='{time:YYYY-MM-DD HH:mm:ss} {level} {message}',
               enqueue=True,  # 支持异步写入
               encoding='utf-8',
               )
    

def clear_dir(fdir, cnt):
    flist = []
    for s in os.listdir(fdir):
        flist.append(fdir + s)

    # 按照创建时间进行排序
    flist.sort(key=lambda x: os.path.getctime(x))
    # print(flist)
    cnt = min(len(flist), cnt)
    for i in range(len(flist) - cnt):
        if os.path.isfile(flist[i]):
            os.remove(flist[i])
        else:
            shutil.rmtree(flist[i])


import os
from PIL import Image
import glob

def clean_corrupted_images(directory):
    # 支持的图像文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
    img_lst = glob.glob(f'{directory}/**/*.*',recursive=True)
    #print(img_lst)
    
    # 遍历目录中的所有文件
    for file_path in img_lst:
        # 检查文件是否是图像文件
        if file_path.lower().endswith(image_extensions):
            try:
                # 尝试打开图像文件
                with Image.open(file_path) as img:
                    # 验证图像是否可以正常加载
                    img.verify()
                    #print(f'read {file_path}')
            except Exception as e:
                # 如果文件无法读取，打印错误并删除
                print(f"Corrupted file: {file_path}, Error: {e}")
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as delete_error:
                    print(f"Failed to delete {file_path}: {delete_error}")

# 使用示例
if __name__ == "__main__":
    current_dir = './data/tiny-imagenet-200/'  # 当前目录
    clean_corrupted_images(current_dir)