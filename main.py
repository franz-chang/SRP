import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["MKL_THREADING_LAYER"] = "GNU"
import torch
import torch.nn as nn
from torchvision.models import (
    resnet34,
    vgg16, densenet121, resnet18,
    vit_b_16  # Vision Transformer (ViT)
)
from torch.nn.functional import softmax
from torchvision import transforms, datasets
import torchvision
from tqdm import tqdm  # 进度条
from max_eigenvalue import power_iteration_max_eigenvalue, compute_max_eigenvalue,hutchinson_max_eigenvalue
import util
import torchattacks
import traceback
import pandas as pd
import torch.nn.functional as F
from torch.autograd import grad
from torchattacks import PGD,CW
import sys
from torch.utils.data import DataLoader
from loguru import logger
import argparse
from medical_data import COVIDRadiographyDataset
from torch.utils.data import random_split
import gc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_examples  = 500

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



def get_dataloaders(model_name,dataset_name,is_test_shuffle = False):
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
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'mnist':
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'tiny-imagenet':
        # Tiny ImageNet需要特殊处理，假设数据已经下载并解压到./data/tiny-imagenet-200
        data_dir = './data/tiny-imagenet-200'
        val_dir = os.path.join(data_dir, 'val')
        
        test_set = datasets.ImageFolder(val_dir, transform=transform_test)
    elif dataset_name == 'cifar100':
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
        test_set.dataset.transform = transform_test  # 测试集通常与验证集相同
     
    test_loader = DataLoader(test_set, batch_size=1, shuffle=is_test_shuffle, num_workers=4)
    
    return test_loader

def get_model(model_name, data_name,attack_type,num_classes):
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
        # 对于tiny-imagenet，使用VGG19
        model = vgg16(weights=None) 
        #model = vgg16(weights=VGG16_Weights.DEFAULT)
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
    

    # 2. 加载权重
    model_path = f'./models/{model_name}_{data_name}_{attack_type}_final.pth'  # 或 _best.pth
    logger.info(f'Loading model from {model_path}...')
    if not os.path.exists(model_path):  
        logger.error(f'Model file {model_path} not found!')
        raise FileNotFoundError(f'Model file {model_path} not found!')

    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    return model.to(device)


#=============================================

def compute_prob_and_gradients(model, x, num_classes):
    """
    计算后验概率 p(y|x) 和梯度矩阵 ∇_x log p(y|x)
    
    Args:
        model: 训练好的模型
        x: 输入张量 (1 x C x H x W)
        num_classes: 类别数 K
        device: 计算设备 ('cuda' 或 'cpu')
    
    Returns:
        Q: 梯度矩阵 (d x K), d = C × H × W
        probs: 后验概率向量 (K,)
    """
    model = model.to(device)
    x = x.to(device).requires_grad_(True)
    
    # 前向传播

    # 前向传播
    logits = model(x)
    probs = softmax(logits, dim=1).squeeze()  # (K,)

    #print(probs.shape)
    
    # 初始化梯度矩阵
    Q = torch.zeros((x.numel(), num_classes), device=device)
    
    # 对每个类别计算梯度
    for k in range(num_classes):

        log_prob = torch.log(probs[k])  # log p(y=k|x)
        log_prob.backward(retain_graph=True)
 
        grad = x.grad.detach().clone().view(-1)  # (d,)
        Q[:, k] = grad
        
        x.grad.zero_()

        del log_prob
        torch.cuda.empty_cache()
        
    
    return Q, probs.detach()


# 主函数：处理整个 CIFAR10 测试集
def estimate_spec(model_name='resnet50',
         data_name='cifar10',attack_type = 'none',n=0):
    
    # classes_num = {
    #     "mnist": 10,
    #     "cifar10": 10,
    #     "tiny-imagenet": 200  # Tiny-ImageNet 有 200 类
    # }
    
    n_classes = dataset_configs[data_name]['num_classes']
    # 加载模型和数据
    model = get_model(model_name,data_name,attack_type, n_classes).to(device)
    model.eval()
    
    logger.info(f'Evaluate spec {model_name} on {data_name} ...')

    all_results = []  # 用于存储特征值结果
    
    testloader = get_dataloaders(model_name,data_name)
     
    for i, (images, labels) in enumerate(testloader):
        #print(images)
        try: 
            images, labels = images.to(device), labels.to(device)

            #print(labels)

            Q, probs = compute_prob_and_gradients(model, images, num_classes=n_classes)
            
            #print(Q, probs)
            
            #当类别数小时，使用
            #lambda_t = compute_max_eigenvalue(Q, probs)

            #lambda_t = power_iteration_max_eigenvalue(Q, probs)

            lambda_t = hutchinson_max_eigenvalue(Q, probs)

            logger.info(f'{n} Batch {i} {model_name} {data_name} lambda {lambda_t:.4f} label {labels.item()}')

            all_results.append((lambda_t, labels.item()))  # 移到 CPU 存储
        except:
            traceback.print_exc()
        
        # for testing
        
        if i >= n_examples - 1:  
            break

    return torch.tensor(all_results)  # 转换为张量


#===================compute sep with black-box settings.=========

def estimate_spec_white(model_name='resnet50',
         data_name='cifar10',attack_type = 'none',n=0):
    
    # classes_num = {
    #     "mnist": 10,
    #     "cifar10": 10,
    #     "tiny-imagenet": 200  # Tiny-ImageNet 有 200 类
    # }
    
    n_classes = dataset_configs[data_name]['num_classes']
    # 加载模型和数据
    model = get_model(model_name,data_name,attack_type, n_classes).to(device)
    model.eval()
    
    logger.info(f'Evaluate spec {model_name} on {data_name} ...')

    all_results = []  # 用于存储特征值结果
    
    testloader = get_dataloaders(model_name,data_name)
     
    for i, (images, labels) in enumerate(testloader):
        #print(images)
        try: 
            images, labels = images.to(device), labels.to(device)

            #print(labels)

            Q, probs = compute_prob_and_gradients(model, images, num_classes=n_classes)
            
            #print(Q, probs)
            
            #当类别数小时，使用
            #lambda_t = compute_max_eigenvalue(Q, probs)

            #lambda_t = power_iteration_max_eigenvalue(Q, probs)

            lambda_t = hutchinson_max_eigenvalue(Q, probs)

            logger.info(f'{n} Batch {i} {model_name} {data_name} lambda {lambda_t:.4f} label {labels.item()}')

            all_results.append((lambda_t, labels.item()))  # 移到 CPU 存储
        except:
            traceback.print_exc()
        
        # for testing
        
        if i >= n_examples - 1:  
            break

    return torch.tensor(all_results)  # 转换为张量



def estimate_spec_black(model_name='resnet50',
         data_name='cifar10',attack_type = 'none',n=0):
    
    # classes_num = {
    #     "mnist": 10,
    #     "cifar10": 10,
    #     "tiny-imagenet": 200  # Tiny-ImageNet 有 200 类
    # }
    
    n_classes = dataset_configs[data_name]['num_classes']
    # 加载模型和数据
    model = get_model(model_name,data_name,attack_type, n_classes).to(device)
    model.eval()
    
    logger.info(f'Evaluate spec black {model_name} on {data_name} ...')

    all_results = []  # 用于存储特征值结果
    
    testloader = get_dataloaders(model_name,data_name)
     
    for i, (images, labels) in enumerate(testloader):
        #print(images)
        try: 
            images, labels = images.to(device), labels.to(device)
            #lambda_t = sequential_blackbox(model,images,m=100)
            lambda_t = parallel_blackbox(model,images,m=100,
                                         chunk_size=100)

            logger.info(f'{n} Batch {i} {model_name} {data_name} lambda {lambda_t:.4f} label {labels.item()}')

            all_results.append((lambda_t, labels.item()))  # 移到 CPU 存储
        except:
            traceback.print_exc()
        
        # for testing
        
        if i >= n_examples - 1:  
            break

    return torch.tensor(all_results)  # 转换为张量



@torch.inference_mode()  # 完全禁用计算图和中间值保存
def sequential_blackbox(logits_fn, x, h=1e-3,m = 50):
    x = x.detach().requires_grad_(False)
    device = x.device
    img_shape = x.shape[1:]  # (C, H, W)
    quad_forms = torch.zeros(m, device=device)  # Store results for each sample
    for i in range(m):
        # Generate a single random direction (shape [C, H, W]) and normalize
        v = torch.randn(*img_shape, device=device)
        v = v.reshape(-1)  # Flatten to [C*H*W]
        v = v / torch.norm(v)  # Normalize (scalar division)
        v = v.reshape(*img_shape)  # Reshape back to [C, H, W]
        
        # Compute log p(y|x + h v) and log p(y|x - h v)
        logits_plus = logits_fn(x + h*v)   # [num_classes]
        logits_minus = logits_fn(x - h*v)  # [num_classes]
        # Compute p(y|x) (same for all v since x is fixed)
        logits = logits_fn(x)  # [num_classes]

        # 立即释放不需要的变量
        del v
        torch.cuda.empty_cache()  # 手动清理CUDA缓存

        log_p_plus = F.log_softmax(logits_plus, dim=-1)  # [num_classes]
        log_p_minus = F.log_softmax(logits_minus, dim=-1)

        # Compute directional derivative: v^T ∇_x log p(y|x) [num_classes]
        directional_derivative = (log_p_plus - log_p_minus) / (2 * h)

        probs = F.softmax(logits, dim=-1)  # [num_classes]

        # Compute quadratic form: sum_y p(y|x) (v^T ∇_x log p(y|x))^2 [scalar]
        quad_forms[i] = torch.sum(probs * (directional_derivative ** 2))

        # 释放中间变量
        del logits_plus, logits_minus, log_p_plus, log_p_minus, directional_derivative, logits, probs
        torch.cuda.empty_cache()

        gc.collect()


    # Return the maximum value across all samples
    return torch.max(quad_forms).item()


@torch.inference_mode()  # 完全禁用计算图和中间值保存
def parallel_blackbox(logits_fn, x, h=1e-3, m=1000, chunk_size=100):
    x = x.detach().to(device)
    img_shape = x.shape[1:]
    max_lambda = -float('inf')
    
    # 预计算 p(y|x)（第一次前向传播）
    with torch.no_grad():
        logits = logits_fn(x)  # (1, num_classes)
        probs = F.softmax(logits, dim=-1)   # (1, num_classes)
    
    # 分块处理随机方向
    for i in range(0, m, chunk_size):
        current_chunk = min(chunk_size, m - i)
        
        # 生成当前块的随机方向（显存仅占用chunk_size）
        v = torch.randn(current_chunk, *img_shape, device=device)
        v = v / torch.norm(v.reshape(current_chunk, -1), dim=1, keepdim=True).view(-1, *([1]*len(img_shape)))
        
        # 计算 x ± h*v（第二、三次前向传播）
        x_plus = x + h * v
        x_minus = x - h * v
        logits_plus = logits_fn(x_plus)    # (chunk_size, num_classes)
        logits_minus = logits_fn(x_minus)  # (chunk_size, num_classes)
        
        # 计算方向导数
        directional_deriv = (F.log_softmax(logits_plus, dim=-1) - 
                            F.log_softmax(logits_minus, dim=-1)) / (2 * h)
        
        # 计算当前块的quad_forms并更新最大值
        quad_forms = torch.sum(probs * (directional_deriv ** 2), dim=-1)
        max_lambda = max(max_lambda, torch.max(quad_forms).item())
        
        # 显存清理（关键！）
        del v, x_plus, x_minus, logits_plus, logits_minus, directional_deriv, quad_forms
        torch.cuda.empty_cache()
    
    return max_lambda


@torch.inference_mode()  # 完全禁用计算图和中间值保存
def parallel_blackbox_big_blocks(logits_fn, x, h=1e-3,m = 100):
    """
    并行计算 Hutchinson 估计的最大特征值。

    Args:
        logits_fn (callable): 输入 x，返回 logits（未归一化的 log p(y|x)）。
        x (torch.Tensor): 输入张量，形状为 (..., d)。
        h (float): 有限差分步长。
        m (int): 并行采样的随机方向数量。

    Returns:
        float: 估计的最大特征值。
    """

    x = x.detach().requires_grad_(False)
    device = x.device

    # 生成 m 个随机方向（形状 (m, *x.shape)），并归一化
    v = torch.randn(m, *x.shape, device=device)
    v = v.reshape(m, -1)  # Flatten all dimensions except batch
    v = v / torch.norm(v, dim=1, keepdim=True)  # Normalize

    img_shape = x.shape[1:]

    v = v.reshape(m, *img_shape)  # Reshape back to original shape
    # 计算所有 v 的方向导数（并行）
    # 扩展 x 到 (m, *x.shape) 以匹配 v 的批次维度
    x_batch = x.expand(m, *img_shape)  # (1, C, H, W) -> (m, C, H, W)

    #print(v.shape)
    #print(x_batch.shape)
    
    # 计算 log p(y|x + h v) 和 log p(y|x - h v)（并行）
    logits_plus = logits_fn(x_batch + h * v)   # (m, num_classes)
    logits_minus = logits_fn(x_batch - h * v)  # (m, num_classes)
    log_p_plus = F.log_softmax(logits_plus, dim=-1)  # (m, num_classes)
    log_p_minus = F.log_softmax(logits_minus, dim=-1)

    # 计算方向导数：v^T ∇_x log p(y|x)（形状 (m, num_classes)）
    directional_derivatives = (log_p_plus - log_p_minus) / (2 * h)

    # 计算 p(y|x)（对所有 v 相同，因为 x 相同）
    logits = logits_fn(x)  # (num_classes,)
    probs = F.softmax(logits, dim=-1)  # (num_classes,)

    # 计算每个 v 的二次型：sum_y p(y|x) (v^T ∇_x log p(y|x))^2（形状 (m,)）
    quad_forms = torch.sum(probs * (directional_derivatives ** 2), dim=-1)  # (m,)
    # 返回最大值
    return torch.max(quad_forms).item()



#============================================
def estimate_lipschitz_with_norm(model,x):
    """
    估计模型的 Lipschitz 常数
    Args:
        model: 训练好的模型
        input: 输入张量 (1 x C x H x W)
        epsilon: 小扰动的大小
    Returns:
        lipschitz_constant: Lipschitz 常数的估计值
    """
    model.eval()
    x = x.to(device).requires_grad_(True)
    
    # 前向传播
    #y = model(x)

    logits = model(x)
    y = softmax(logits, dim=1)  # (K,)
    #print(logits.shape,y.shape)

    grad_norms = []
            
    for k in range(y.size(1)):
        grad_output = torch.zeros_like(y)
        grad_output[:, k] = 1.0
        grad_input = grad(y, x, grad_outputs=grad_output, 
                        retain_graph=True, create_graph=False)[0]
        grad_norms.append(grad_input.norm(p=2, dim=(1,2,3)))
    
    # 取最大梯度范数作为当前batch的Lipschitz估计
    max_grad_norm = torch.max(torch.stack(grad_norms)).item()
    return max_grad_norm


def estimate_lips(model_name='resnet50', data_name='cifar10',attack_type='none', n=0):
    """
    主函数：处理整个数据集并估计Lipschitz常数 (GPU兼容版本)
    Args:
        model_name: 模型名称
        data_name: 数据集名称
        device: 计算设备 ('cuda' 或 'cpu')
    Returns:
        包含所有Lipschitz常数估计和对应标签的张量
    """

    # 类别数量映射
    # classes_num = {
    #     "mnist": 10,
    #     "cifar10": 10,
    #     "tiny-imagenet": 200,  # Tiny-ImageNet 有 200 类
    #     'cifar100': 100,
    # } 
    #n_classes = classes_num[data_name]

    n_classes = dataset_configs[data_name]['num_classes']
    
    # 加载模型和数据
    model = get_model(model_name,data_name,attack_type, n_classes).to(device)
    model.eval()
    
    logger.info(f'Evaluate Lipschitz constant for {model_name} on {data_name}   using {device}...')

    batch_size = 1
    all_results = []  # 用于存储特征值结果
    
    # 加载数据集 (假设load_dataset函数已定义)
    testloader = get_dataloaders(model_name,data_name)
     
    for i, (images, labels) in enumerate(testloader):
        try: 
            # 确保数据和模型在同一设备上
            images, labels = images.to(device), labels.to(device)
            
            # 估计Lipschitz常数
            lip_const = estimate_lipschitz_with_norm(model, images)
            
            logger.info(f'{n} Batch {i}: {model_name} on {data_name} - Lipschitz constant: {lip_const:.4f}')
            
            # 将结果移到CPU存储 (减少GPU内存使用)
            all_results.append((lip_const, labels.cpu().item()))
            
        except Exception as e:
            logger.info(f"Error processing batch {i}:")
            traceback.print_exc()
        
        # 测试用 - 可以取消注释来只处理少量样本
        if i >= n_examples - 1:
            break
    
    # 返回结果张量 (在CPU上)
    return torch.tensor(all_results)

# attack_type确定需要装载哪个模型
#===========================================================

def estimate_pgd(model_name='resnet50', data_name='cifar10', 
                    attack_type = 'cw',n = 0):
    """
    评估模型在PGD攻击下的鲁棒性 (GPU兼容版本)
    Args:
        model_name: 模型名称
        data_name: 数据集名称
        device: 计算设备 ('cuda' 或 'cpu')
    Returns:
        包含所有评估结果和对应标签的张量
    """

    # 类别数量映射
    # classes_num = {
    #     "mnist": 10,
    #     "cifar10": 10,
    #     "tiny-imagenet": 200  # Tiny-ImageNet 有 200 类
    # } 
    # n_classes = classes_num[data_name]

    n_classes = dataset_configs[data_name]['num_classes']
    
    # 加载模型并移动到指定设备
    model = get_model(model_name,data_name,attack_type, n_classes).to(device)
    model.eval()
    
    logger.info(f'Evaluating pgd attack of {model_name} on {data_name} using {device}...')

    batch_size = 1
    all_results = []  # 用于存储评估结果
    
    # 加载数据集
    testloader = get_dataloaders(model_name,data_name)
    
    # 初始化PGD攻击
    #attack_pgd = PGD(model)  # 确保攻击方法也在正确设备上
    
    attack_pgd = PGD(model, eps=pgd_params['eps'], alpha=pgd_params['alpha'], 
                        steps=pgd_params['steps'], random_start=pgd_params['random_start'])

    for i, (images, labels) in enumerate(testloader):
        try: 
            # 确保数据和标签在正确设备上
            images, labels = images.to(device), labels.to(device)
            
            x = attack_pgd(images, labels)

            # 评估对抗样本
            with torch.no_grad():
                pred_attack = model(x).argmax(1)
                pred_std = model(images).argmax(1)

                result_attack = (pred_attack == labels).float().mean().item()  # 计算正确率
                result_std = (pred_std == labels).float().mean().item()
            

            logger.info(f'{n} Batch {i}: {model_name} on {data_name} - attack type: {attack_type} pgd: {result_attack} std: {result_std}')
            
            # 将结果和标签移到CPU存储
            all_results.append((result_attack,result_std))
            
        except Exception as e:
            logger.info(f"Error processing batch {i}:")
            traceback.print_exc()
        
        # 测试用 - 可以取消注释来只处理少量样本
        if i >= n_examples - 1:  
            break
    
    # 返回结果张量 (在CPU上)
    return torch.tensor(all_results)


def estimate_cw(model_name='resnet50', data_name='cifar10', 
                    attack_type = 'cw',n = 0):
    """
    评估模型在PGD攻击下的鲁棒性 (GPU兼容版本)
    Args:
        model_name: 模型名称
        data_name: 数据集名称
        device: 计算设备 ('cuda' 或 'cpu')
    Returns:
        包含所有评估结果和对应标签的张量
    """

    # 类别数量映射
    # classes_num = {
    #     "mnist": 10,
    #     "cifar10": 10,
    #     "tiny-imagenet": 200  # Tiny-ImageNet 有 200 类
    # }
    # n_classes = classes_num[data_name]

    n_classes = dataset_configs[data_name]['num_classes']
    
    # 加载模型并移动到指定设备
    model = get_model(model_name,data_name,attack_type, n_classes).to(device)
    model.eval()
    
    logger.info(f'Evaluating attack of {model_name} on {data_name} using {device}...')

    batch_size = 1
    all_results = []  # 用于存储评估结果
    
    # 加载数据集
    testloader = get_dataloaders(model_name,data_name)
    
    # 初始化PGD攻击
    #attack_pgd = PGD(model)  # 确保攻击方法也在正确设备上
    
    attack_cw = CW(model, c=cw_params['c'], steps=cw_params['steps'], lr=cw_params['lr'])

    for i, (images, labels) in enumerate(testloader):
        try: 
            # 确保数据和标签在正确设备上
            images, labels = images.to(device), labels.to(device)

            x = attack_cw(images, labels)

            # 评估对抗样本
            with torch.no_grad():
                pred_attack = model(x).argmax(1)
                pred_std = model(images).argmax(1)

                result_attack = (pred_attack == labels).float().mean().item()  # 计算正确率
                result_std = (pred_std == labels).float().mean().item()
            

            logger.info(f'{n} Batch {i}: {model_name} on {data_name} - attack type: {attack_type} cw: {result_attack} std: {result_std}')
            
            # 将结果和标签移到CPU存储
            all_results.append((result_attack,result_std))
            
        except Exception as e:
            logger.info(f"Error processing batch {i}:")
            traceback.print_exc()
        
        # 测试用 - 可以取消注释来只处理少量样本
        if i >= n_examples - 1:  
            break
    
    # 返回结果张量 (在CPU上)
    return torch.tensor(all_results)

#====================================
def clever_score(model, x, true_label, num_classes, num_samples=100, 
                 sigma=0.1, norm='2'):
    """
    计算CLEVER分数 (Cross Lipschitz Extreme Value for nEtwork Robustness)
    
    Args:
        model: 目标模型
        x: 输入样本 (1 x C x H x W)
        true_label: 真实标签
        num_classes: 类别数量
        num_samples: 采样次数
        sigma: 高斯噪声的标准差
        norm: 范数类型 ('1', '2', 'inf')
        device: 计算设备
        
    Returns:
        clever_score: 估计的CLEVER分数
    """
    model = model.to(device)
    x = x.to(device)
    true_label = true_label.to(device)
    
    # 获取输入维度
    input_shape = x.shape
    dim = x.numel()
    
    # 初始化存储梯度的列表
    gradients = []
    
    # 采样过程
    for _ in range(num_samples):
        # 生成随机方向向量
        if norm == '2':
            # 对于L2范数，从单位球面均匀采样
            v = torch.randn_like(x)
            v = v / v.norm(p=2)
        elif norm == 'inf':
            # 对于L∞范数，从{-1,1}^d均匀采样
            v = torch.randint_like(x, low=0, high=2).float() * 2 - 1
            v = v / v.norm(p=float('inf'))
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        
        # 添加高斯噪声
        delta = sigma * v
        x_perturbed = x + delta
        
        # 计算梯度
        x_perturbed = x_perturbed.requires_grad_(True)
        logits = model(x_perturbed)
        probs = F.softmax(logits, dim=1)
        
        # 计算目标类和其他类的概率差
        p_true = probs[0, true_label]
        p_other = torch.sum(probs) - p_true  # 其他类的总概率
        
        # 计算梯度
        grad_output = torch.zeros_like(logits)
        grad_output[0, true_label] = 1.0
        grad_output[0, :] -= 1.0 / num_classes  # 中心化
        
        grad_input = grad(logits, x_perturbed, grad_outputs=grad_output,
                          retain_graph=False, create_graph=False)[0]
        
        # 计算梯度范数
        if norm == '2':
            grad_norm = grad_input.norm(p=2)
        elif norm == 'inf':
            grad_norm = grad_input.norm(p=1)  # 对于L∞，使用L1范数近似
        else:
            grad_norm = grad_input.norm(p=float(norm))
        
        gradients.append(grad_norm.item())
    
    # 使用极值理论估计CLEVER分数
    gradients = torch.tensor(gradients)
    
    # 使用Weibull分布拟合极值
    # 这里简化处理，直接取最小值作为估计
    clever_score = torch.min(gradients).item()
    
    return clever_score

def estimate_clever(model_name='resnet50', data_name='cifar10', 
                attack_type = 'none',
                   num_samples=100, sigma=0.1, norm='2', n = 0):
    """
    主函数：评估模型在数据集上的CLEVER分数
    Args:
        model_name: 模型名称
        data_name: 数据集名称
        num_samples: 每个样本的采样次数
        sigma: 高斯噪声标准差
        norm: 范数类型 ('1', '2', 'inf')
    Returns:
        包含所有CLEVER分数和对应标签的张量
    """
    
    # 类别数量映射
    # classes_num = {
    #     "mnist": 10,
    #     "cifar10": 10,
    #     "tiny-imagenet": 200  # Tiny-ImageNet 有 200 类
    # }
    # n_classes = classes_num[data_name]

    n_classes = dataset_configs[data_name]['num_classes'] 
    # 加载模型和数据
    model = get_model(model_name,data_name,attack_type, n_classes).to(device)
    model.eval()
    
    logger.info(f'Evaluate CLEVER score for {model_name} on {data_name} using {device}...')
    all_results = []  # 用于存储结果
    
    # 加载数据集
    testloader = get_dataloaders(model_name,data_name)
     
    for i, (images, labels) in enumerate(testloader):
        try: 
            images, labels = images.to(device), labels.to(device)
            
            # 计算CLEVER分数
            score = clever_score(model, images, labels, n_classes, 
                               num_samples=num_samples, sigma=sigma, norm=norm)
            
            logger.info(f'{n} Batch {i}: {model_name} on {data_name} - CLEVER score: {score:.4f}')
            
            # 将结果移到CPU存储
            all_results.append((score, labels.cpu().item()))
            
        except Exception as e:
            logger.info(f"Error processing batch {i}:")
            traceback.print_exc()
        
        # 测试用 - 可以取消注释来只处理少量样本
        if i >= n_examples - 1: 
            break
    
    # 返回结果张量 (在CPU上)
    return torch.tensor(all_results)


def main(eval_method,
         model_name,
         data_name,
         attack_type = 'none',seed_n = 0):

    # method description ['spec','lips','attack','clever']

    # 示例：使用 ResNet50 处理 CIFAR10 前 100 个样本

    # model_lst = [
    #     'vit_b_16',
    #     'resnet18',
    #     'vgg16',
    #     'densenet121',
    # ]
    
    # data_lst = [
    #     'cifar10',
    #     # 'mnist',
    #     # 'tiny-imagenet',
    # ]

    # print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # print("Torch current device:", torch.cuda.current_device())
    
    result_dir = f'./{attack_type}-results-5/'
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)  

    #确保所有的设置都相同
    util.set_seed((seed_n + 1)*17)
    fname = f'{result_dir}{model_name}-{data_name}-{eval_method}-{seed_n}.csv'
    if eval_method == 'spec':
        result_lst = estimate_spec_white(
            model_name=model_name,
            data_name=data_name,
            attack_type=attack_type,
            n=seed_n)
        df = pd.DataFrame(result_lst.numpy(), columns=['lambda', 'label'])
    elif eval_method == 'specb':
        result_lst = estimate_spec_black(
            model_name=model_name,
            data_name=data_name,
            attack_type=attack_type,
            n=seed_n)
        df = pd.DataFrame(result_lst.numpy(), columns=['lambda', 'label'])
    
    elif eval_method == 'lips':  
        result_lst = estimate_lips(
            model_name=model_name,
            data_name=data_name,
            attack_type=attack_type,
            n=seed_n)
        df = pd.DataFrame(result_lst.numpy(), columns=['lips', 'label'])
    elif eval_method == 'clever':
        result_lst = estimate_clever(
            model_name=model_name,
            data_name=data_name,
            attack_type=attack_type,
            num_samples=2,  # 可以根据需要调整
            sigma=5,       # 噪声标准差
            norm='2',         # 使用L2范数
            n=seed_n
        )
        df = pd.DataFrame(result_lst.numpy(), columns=['clever', 'label']) 
    elif eval_method == 'cw':
        result_lst = estimate_cw(
            model_name=model_name,
            data_name=data_name,
            attack_type=attack_type,
            n=seed_n)
        df = pd.DataFrame(result_lst.numpy(), columns=['cw','std'])
    elif eval_method == 'pgd':
        result_lst = estimate_pgd(
            model_name=model_name,
            data_name=data_name,
            attack_type=attack_type,
            n=seed_n)
        df = pd.DataFrame(result_lst.numpy(), columns=['pgd','std'])
    
    df.to_csv(fname, index=False)



if __name__ == "__main__":

    util.init_logger()
   
    parser = argparse.ArgumentParser(description='Evaluate the model on the dataset with the attack type')

    parser.add_argument('--eval',help='lips,clever,spec,cw,pgd',default='spec',choices=['lips','clever','spec','specb','cw','pgd']) 

    parser.add_argument('--data',help='cifar10,mnist,tiny-imagenet',default='cifar10')

    parser.add_argument('--model',help='vit_b_16,resnet18,vgg16,densenet121',default='resnet18',choices=['vit_b_16','resnet18','vgg16','densenet121'])

    parser.add_argument('--attack',help='cw,pgd,none',default='cw',choices=['pgd','cw','none'])

    parser.add_argument('--seed_n',help='0,1,2,3 for seed',type=int,default=0)

    args = parser.parse_args()

    main(args.eval,
         args.model,
         args.data,
         args.attack,
         args.seed_n)
    


    