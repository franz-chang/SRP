import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os


class COVIDRadiographyDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (str): 数据集根目录（包含各个类别的子文件夹）
            split (str): 训练/验证/测试集划分
            transform (callable): 数据增强
        """
        self.root_dir = root
        self.transform = transform
        self.classes = ['COVID', 'Normal', 'Viral Pneumonia', 'Lung_Opacity']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 收集样本路径和标签
        self.samples = []
        for cls in self.classes:
            cls_dir = f'{self.root_dir}/{cls}/images/'
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.png'):
                    self.samples.append((
                        os.path.join(cls_dir, img_name),
                        self.class_to_idx[cls]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # 确保3通道
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
