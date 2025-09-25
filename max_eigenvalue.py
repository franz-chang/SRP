import torch
import numpy as np
import time
import random
import util


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_max_eigenvalue(Q, Lambda_diag):
    """
    Compute the largest eigenvalue of B = Λ^{1/2} Q^T Q Λ^{1/2},
    where Q is d x K, K << d, and Λ is diagonal.
    
    Args:
        Q: (d, K) tensor
        Lambda_diag: (K,) tensor, diagonal elements of Λ
    
    Returns:
        max_eigenvalue: scalar tensor
    """
    device = Q.device
    K = Q.shape[1]
    
    # Compute Q^T Q (K x K matrix)
    Q_T_Q = Q.T @ Q  # (K, K)
    
    # Compute Λ^{1/2} Q^T Q Λ^{1/2}
    Lambda_sqrt = Lambda_diag.sqrt()  # (K,)
    B = Lambda_sqrt.view(-1, 1) * Q_T_Q * Lambda_sqrt.view(1, -1)  # (K, K)
    
    # Compute eigenvalues of B (K x K matrix)
    eigenvalues = torch.linalg.eigvalsh(B)  # Real symmetric => use eigvalsh
    
    # Return the largest eigenvalue
    return eigenvalues.max()


def power_iteration_max_eigenvalue(Q, Lambda, max_iter=100):
    """
    使用幂迭代法计算 A = Q Lambda Q^T 的最大特征值
    
    """
    # 移至GPU并确保数据类型一致
    Q = Q.clone().detach().to(device=device, dtype=torch.float32)
    Lambda = Lambda.clone().detach().to(device=device, dtype=torch.float32)
    
    max_idx = torch.argmax(Lambda)
    x = Q[:,max_idx] 
    x = x / torch.norm(x)
    lambda_prev = Lambda[max_idx].clone().detach().to(device=device, dtype=torch.float32)

    tol = 1e-6
    total_iter = max_iter
    for i in range(max_iter):
        # 高效计算 A x = Q Lambda Q^T x
        y = torch.matmul(Q.T, x)          # Q^T x: O(nd)
        z = Lambda * y                     # Lambda Q^T x: O(d)
        y_t = torch.matmul(Q, z)           # Q (Lambda Q^T x): O(nd)
        
        lambda_t = torch.dot(x, y_t)       # Rayleigh 商
        
        # 检查收敛性（相对误差）
        if torch.abs(lambda_t - lambda_prev) < tol * torch.abs(lambda_t):
            total_iter = i
            break
        
        lambda_prev = lambda_t
        x = y_t / torch.norm(y_t)          # 归一化
    
    return lambda_t,total_iter



def hutchinson_max_eigenvalue(Q, Lambda, num_iter=100, is_normal = True):
    """
    向量化Hutchinson算法计算 B = Lambda^{1/2} Q^T Q Lambda^{1/2} 的最大特征值
    
    参数:
        Q (torch.Tensor): 形状为 (n, d) 的矩阵
        Lambda (torch.Tensor): 形状为 (d,) 的对角阵元素
        num_iter (int): 迭代次数
        device (str): 设备 ('cuda' 或 'cpu')
    
    返回:
        max_eigenvalue (float): 最大特征值的估计值
    """
    # 移至GPU并确保数据类型一致
    Q = Q.clone().detach().to(device=device, dtype=torch.float32)
    Lambda = Lambda.clone().detach().to(device=device, dtype=torch.float32)
    
    d = Q.size(1)
    Lambda_sqrt = torch.sqrt(Lambda)  # Lambda^{1/2}
    
    if is_normal:
        # 一次性生成所有随机向量 z ~ N(0, I_d), 形状 (d, num_iter)
        Z = torch.randn(d, num_iter, device=device)
    else:
        # 生成 Rademacher 向量 (z_i ∈ {+1, -1})
        Z = torch.randint(0, 2, (d, num_iter), device=device).float() * 2 - 1  # 转换为 {-1, +1}
        #print(Z)
    
    # 计算 BZ = Lambda^{1/2} Q^T Q Lambda^{1/2} Z [等效于 (d, num_iter)]
    Q_Lambda_sqrt_Z = Q @ (Lambda_sqrt.unsqueeze(1) * Z)  # Q (Lambda^{1/2} Z), 形状 (n, num_iter)
    BZ = Lambda_sqrt.unsqueeze(1) * (Q.t() @ Q_Lambda_sqrt_Z)  # Lambda^{1/2} Q^T Q Lambda^{1/2} Z
    
    # 计算所有瑞利商 z^T B z / z^T z [形状 (num_iter,)]
    #rayleigh_quotients = (Z * BZ).sum(dim=0) / (Z * Z).sum(dim=0)
    
    if is_normal:
        rayleigh_quotients = (Z * BZ).sum(dim=0) / (Z * Z).sum(dim=0)
    else:
        rayleigh_quotients = (Z * BZ).sum(dim=0) / d
    
    # 取最大值作为估计
    max_eigenvalue = rayleigh_quotients.max().item()

    del Q,Lambda,Lambda_sqrt,Z,Q_Lambda_sqrt_Z,BZ,rayleigh_quotients
    torch.cuda.empty_cache()
    return max_eigenvalue


def hutchinson_max_eigenvalue_aux(Q, Lambda, num_samples=1000, is_normal=True):
    """
    Hutchinson 算法并行计算 A = Q Lambda Q^T 的最大特征值（GPU 加速）
    
    Args:
        Q: (n, k) 矩阵
        Lambda: (k,) 对角矩阵的对角元素
        num_samples: 采样次数
        device: "cuda" 或 "cpu"
    Returns:
        最大特征值的估计
    """

 # 确保输入是 PyTorch Tensor 并移到 GPU
    Q = Q.clone().detach().to(device=device, dtype=torch.float32)
    Lambda = Lambda.clone().detach().to(device=device, dtype=torch.float32)
    n, k = Q.shape
    
    if is_normal:
        # 生成随机向量 z (num_samples, n)
        z = torch.randn(num_samples, n, device=device)  # 直接在 GPU 上生成
    else:
        # 生成Rademacher随机向量（±1）
        z = torch.randint(0, 2, (num_samples, n), device=device).float() * 2 - 1

    # 计算 y = Q^T z (num_samples, k)
    y = torch.matmul(z, Q)  # (num_samples, k)

    # 计算 z^T A z = y^T Lambda y (num_samples,)
    zTAz = torch.sum(y * Lambda * y, dim=1)  # 逐元素乘法和求和
    
    if is_normal:
        # 计算 z^T z (num_samples,)
        zTz = torch.sum(z * z, dim=1)
    else:
        zTz = num_samples

    # 计算 Rayleigh 商 (num_samples,)
    rayleigh_quotients = zTAz/zTz

    # 取最大值作为估计
    max_eigenvalue = torch.max(rayleigh_quotients).item()
    return max_eigenvalue
    
    
    
    
    
    
    
    

    