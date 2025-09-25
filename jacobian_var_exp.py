
import torch
import torch.nn as nn
from torch.autograd import grad
from torchvision.models import (
    ResNet50_Weights,ResNet18_Weights,
    VGG16_Weights,VGG19_Weights,
    DenseNet121_Weights,DenseNet169_Weights,
    ViT_B_16_Weights,
    resnet50, vgg16,vgg19, densenet121,densenet169,resnet18,
    vit_b_16  # Vision Transformer (ViT)
)
import util
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cpu")

class FisherInformationValidator:
    def __init__(self, model):
        self.model = model.to(device)
        self.device = device
        
    def compute_softmax_jacobian(self, x):
        """计算softmax输入f(x)对x的Jacobian矩阵 J = df/dx"""
        x = x.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.model(x)  # 获取softmax前的输出
            if f.ndim == 1:
                f = f.unsqueeze(0)
            batch_size, num_classes = f.shape
            
            # 计算Jacobian矩阵 [batch_size, num_classes, *x_shape]
            J = torch.zeros((batch_size, num_classes, *x.shape[1:]), device=self.device)
            for k in range(num_classes):
                grad_output = torch.zeros_like(f)
                grad_output[:, k] = 1.0
                J[:, k] = grad(f, x, grad_outputs=grad_output, 
                               retain_graph=True, create_graph=False)[0]
        return J, f
    
    def compute_fim(self, J, f):
        """计算Fisher信息矩阵 F = Cov(J) = E[J^T J] - E[J]^T E[J]"""
        # 展平Jacobian矩阵 [batch_size, num_classes, input_dim]
        J_flat = J.flatten(start_dim=2)  
        input_dim = J_flat.shape[-1]
        
        # 计算经验FIM
        J_mean = J_flat.mean(dim=0)  # [num_classes, input_dim]
        F = torch.einsum('bki,bkj->ij', J_flat, J_flat) / J_flat.size(0) - J_mean.T @ J_mean
        
        # 理论FIM (对于softmax分类器)
        prob = torch.softmax(f, dim=1)
        Cov_f = torch.diag_embed(prob) - torch.einsum('bi,bj->bij', prob, prob)
        F_theory = torch.einsum('bki,bkl,blj->ij', 
                               J_flat, Cov_f, J_flat) / J_flat.size(0)
        
        return F, F_theory
    
    def verify_theorem(self, x):
        """验证F = Cov(J)定理"""
        J, f = self.compute_softmax_jacobian(x)
        F_empirical, F_theory = self.compute_fim(J, f)
        
        # 计算相对误差
        error = torch.norm(F_empirical - F_theory, p='fro') / torch.norm(F_theory, p='fro')

        return error.item()


def jacobian_variance_exp():
    model_name = 'resnet50'
    f = open('jacobian_variance.txt','w')
    #model = get_model(model_name,100)
    
    # 1. 定义模型 (示例: 简单CNN + softmax)
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.fc = nn.Linear(16*30*30, 10)  # 假设输入32x32，输出10类
            
        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)  # 返回softmax前的logits
    
    model = TestModel()
    validator = FisherInformationValidator(model)
    batch_lst = [32,64,128,256,512,1024]
    n_repeats = 10
    for batch_size in batch_lst:
        util.set_seed(17) 
        avg_error = 0
        for i in range(n_repeats):
            # 3. 生成测试数据 (batch_size=32)
            x_test = torch.randn(batch_size, 3, 32, 32, device=device)
            avg_error += validator.verify_theorem(x_test)
        
        avg_error /= n_repeats
        
        print(f'{batch_size} {avg_error:.4f}\n')
        f.write(f'{batch_size} {avg_error:.4f}\n')
    
    f.close()
        

if __name__ == "__main__":
    jacobian_variance_exp()
    
    # data = np.loadtxt('jacobian_variance.txt')
    # plt.plot(data[:,0],data[:,1])
    # plt.xlabel('Sampling Size')
    # plt.ylabel('Relative Error') 
    # plt.savefig('jacobian_variance.pdf')
    
    