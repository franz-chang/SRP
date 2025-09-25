import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
import sys
from main import get_dataloaders,get_model,dataset_configs,compute_prob_and_gradients

from loguru import logger
import util
from torch.nn import functional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_examples  = 500


# 定义 KL 散度计算
# def kl_divergence(p, q):
#     return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)


def kl_divergence(p, q, eps=1e-8):
    """数值稳定的KL散度计算"""
    p = p.clone() + eps
    q = q.clone() + eps
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)

# 定义扰动生成（l∞ ≤ 8/255）
def generate_perturbation(x, epsilon=8/255):
    delta = torch.rand_like(x) * 2 * epsilon - epsilon  # 均匀采样 [-ε, ε]
    return delta.clamp(-epsilon, epsilon)  # 确保 l∞ ≤ ε


# 计算近似误差
def compute_approximation_error(x, delta,num_classes):
    x_perturbed = x + delta
    
    # 计算 p(y|x) 和 p(y|x')
    with torch.no_grad():
        logits = model(x)
        logits_perturbed = model(x_perturbed)
        p = functional.softmax(logits, dim=-1)
        p_perturbed = functional.softmax(logits_perturbed, dim=-1)
    
    # 计算 KL(p(y|x) || p(y|x'))
    kl = kl_divergence(p, p_perturbed)

    #print('compute KL divergence ...')
    
    # 计算 Fisher 信息矩阵 F(x)
    Q, probs = compute_prob_and_gradients(model, x, num_classes)
    
    #print('compute the Q matrix ...')    
    # 计算二阶泰勒近似 1/2 δ^T F δ
    delta_flat = delta.view(-1)

    # print(Q.size())
    # print(delta_flat.size())

    Qx = delta_flat@Q  # 或 torch.matmul(Q, x)
    taylor_approx = 0.5*torch.sum(probs * Qx ** 2)
        
    # 计算近似误差 |KL - Taylor|
    #error = torch.abs(kl - taylor_approx)
    error = torch.abs(kl - taylor_approx)/kl
    return error.item()

if __name__ == '__main__':
    util.init_logger('kl_fim_data')
    util.set_seed(42)

    data_name = sys.argv[1]
    model_name = sys.argv[2]

    #model_name = 'resnet18'

    attack_type = 'cw'

    n_classes = dataset_configs[data_name]['num_classes']
    # 加载模型和数据
    model = get_model(model_name,data_name,attack_type, n_classes).to(device)
    model.eval()
    
    logger.info(f'Evaluate abs(Kl-FIM) {model_name} on {data_name} ...')
    testloader = get_dataloaders(model_name,data_name,is_test_shuffle=True)


    # 在测试集上计算平均误差
    errors = []
    for i, (x, y) in enumerate(tqdm(testloader)):
        if i >= n_examples:  # 仅用 100 个样本估计
            break

        x = x.to(device)
        delta = generate_perturbation(x)
        error = compute_approximation_error(x, delta,n_classes)
        errors.append(error)

        logger.info(f'iter {i} error {error:.6f}')

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    logger.info(f"avg: {mean_error:.6f} std: {std_error} max: {max_error} min: {min_error}")



