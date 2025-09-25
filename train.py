
import os
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm
from torchvision.models import (
    ResNet50_Weights, ResNet18_Weights,
    VGG16_Weights, ResNet34_Weights,
    DenseNet121_Weights, DenseNet169_Weights,VGG19_Weights,
    ViT_B_16_Weights, resnet34,
    resnet50, vgg16,vgg19, densenet121, densenet169, resnet18,
    vit_b_16  # Vision Transformer (ViT)
)
import util
import sys
import os
from tqdm import tqdm
import timm  # 用于加载Vision Transformer模型
from loguru import logger
from medical_data import COVIDRadiographyDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import timm  # 用于加载Vision Transformer模型
from torchattacks import PGD, CW  # 导入对抗攻击库
import argparse
from torch.utils.data import random_split

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
batch_size = 32
num_epochs = 10

pgd_params = {
    'eps': 8/255,
    'alpha': 2/255,
    'steps': 20,
    'random_start': True
}
cw_params = {
    'c': 1,
    'steps': 20,
    'lr': 0.01
}

# 数据集配置
dataset_configs = {
    'cifar10': {
        'num_classes': 10,
        'input_size': 224,  # 修改为224以适配ViT
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    },
    'cifar100': {
        'num_classes': 100,
        'input_size': 224,  # 修改为224以适配ViT
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    },
    'medical': {
        'num_classes':4,
        'input_size': 224,
        'mean': (0.485, 0.456, 0.406),  # ImageNet标准（若用预训练模型）
        'std': (0.229, 0.224, 0.225),
    },
    'mnist': {
        'num_classes': 10,
        'input_size': 224,  # 修改为224以适配ViT
        'mean': (0.1307,),
        'std': (0.3081,)
    },
    'tiny-imagenet': {
        'num_classes': 200,
        'input_size': 224,  # 修改为224以适配ViT
        'mean': (0.4802, 0.4481, 0.3975),
        'std': (0.2302, 0.2265, 0.2262)
    }
}

def get_dataloaders(dataset_name, model_name):
    """获取指定数据集的数据加载器"""
    config = dataset_configs[dataset_name]
    
    # 对于ViT模型，强制使用224x224输入
    if model_name == 'vit_b_16':
        input_size = 224
    else:
        input_size = config['input_size']
    
    # 数据预处理
    if dataset_name == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])
        transform_test = transform_train
    else:
        transform_train = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size, padding=4) if input_size > 32 else transforms.RandomCrop(input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])
    
    # 加载数据集
    if dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'mnist':
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar100':
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'medical':
        #https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

        # 创建Dataset和DataLoader
        full_dataset = COVIDRadiographyDataset(
            root='./data/medical_data/',
            transform=None)
        
        # 按比例划分（例如 70%/30%）
        train_size = int(0.7 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        logger.info(f'load train {train_size} test {test_size} ...')
        
        train_set, test_set = random_split(
            full_dataset, 
            [train_size,test_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子
        )
        train_set.dataset.transform = transform_train
        test_set.dataset.transform = transform_test  # 测试集通常与验证集相同
    elif dataset_name == 'tiny-imagenet':
        # Tiny ImageNet需要特殊处理，假设数据已经下载并解压到./data/tiny-imagenet-200
        data_dir = './data/tiny-imagenet-200'
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        train_set = datasets.ImageFolder(train_dir, transform=transform_train)
        test_set = datasets.ImageFolder(val_dir, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, config['num_classes']

def get_model(model_name, data_name,num_classes):
    """获取指定模型并修改最后一层以适应分类任务"""
    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            model = resnet18(weights=None)
        elif model_name == 'resnet34':
            model = resnet34(weights=None)
        
        # 修改最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'vgg16':
        model = vgg16(weights=None)
        
        # 修改最后一层
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet121':
        model = densenet121(weights=None)
        
        # 修改最后一层
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'vit_b_16':
        model = vit_b_16(weights=None)
            
        #修改最后一层
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)

    return model.to(device)

def train_model(model, train_loader, criterion, optimizer, epoch, adv_method=None):
    """训练模型，支持正常训练和对抗训练"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 根据选择的对抗训练方法初始化攻击器
    if adv_method == 'pgd':
        attack = PGD(model, eps=pgd_params['eps'], alpha=pgd_params['alpha'], 
                    steps=pgd_params['steps'], random_start=pgd_params['random_start'])
    elif adv_method == 'cw':
        attack = CW(model, c=cw_params['c'], steps=cw_params['steps'], lr=cw_params['lr'])
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 对抗训练：生成对抗样本
        if adv_method in ['pgd', 'cw']:
            adv_inputs = attack(inputs, labels)
            inputs = torch.cat([inputs, adv_inputs], dim=0)
            labels = torch.cat([labels, labels], dim=0)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def test_model(model, test_loader, criterion):
    """测试模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def main(model_name, data_name, adv_train_method = 'none'):
    # 确保保存模型的目录存在
    os.makedirs('./models', exist_ok=True)

    # 获取数据加载器
    train_loader, test_loader, num_classes = get_dataloaders(data_name, model_name)

    logger.info(f"----- Training model: {model_name} on {data_name} with classes: {num_classes} -----")
    logger.info(f"Adversarial training method: {adv_train_method if adv_train_method != 'none' else 'None'}")
    
    # 获取模型
    model = get_model(model_name, data_name, num_classes)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    
    # 训练和测试循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        # 训练（根据设置选择是否使用对抗训练）
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, epoch, 
            adv_method=adv_train_method if adv_train_method != 'none' else None
        )
        
        # 测试
        test_loss, test_acc = test_model(model, test_loader, criterion)

        logger.info(f'Epoch {epoch + 1}/{num_epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = f'./models/{model_name}_{data_name}_{adv_train_method}_best.pth'
            torch.save(model.state_dict(), model_path)
            logger.info(f'Best model saved at: {model_path} with accuracy: {best_acc:.2f}%')
    
    # 保存最终模型
    final_model_path = f'./models/{model_name}_{data_name}_{adv_train_method}_final.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'Final model saved at: {final_model_path}')



if __name__ == '__main__':
    util.init_logger()
    util.set_seed(42)
    
    parser = argparse.ArgumentParser(description='Evaluate the model on the dataset with the attack type')

    parser.add_argument('--data',help='cifar10,mnist,tiny-imagenet',default='cifar10')

    parser.add_argument('--model',help='vit_b_16,resnet18,vgg16,densenet121',default='resnet18',choices=['vit_b_16','resnet18','vgg16','densenet121'])

    parser.add_argument('--attack',help='cw,pgd,none',default='cw')

    args = parser.parse_args()
    
    #main(model_name, data_name, adv_train_method = 'none'):

    main(args.model,
         args.data,
         args.attack)
    

