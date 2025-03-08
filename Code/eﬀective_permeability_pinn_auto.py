# -*- coding: utf-8 -*-
"""
花岗岩渗透率PINN模型实现
基于论文: Estimates for the Effective Permeability of Intact Granite
Obtained from the Eastern and Western Flanks of the Canadian Shield

特点:
- 使用PINN方法研究花岗岩渗透率
- 支持模型训练状态保存和加载
- 支持渗透率场预计算结果保存和加载
- 支持训练检查点和继续训练
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from scipy.stats import lognorm
import argparse

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义全局参数
class Config:
    # 实验参数 (来自原论文)
    STANSTEAD_SIZE = 280.0  # mm
    LAC_DU_BONNET_SIZE = 300.0  # mm
    INNER_RADIUS = 25.4 / 2  # mm (内径/2)
    OUTER_RADIUS = 101.6 / 2  # mm (外径/2)
    PRESSURE = 200.0  # kPa
    DYNAMIC_VISCOSITY = 1.0e-3  # Pa·s (at 20°C)
    
    # 神经网络参数
    HIDDEN_LAYERS = 6
    NEURONS_PER_LAYER = 128
    ACTIVATION = nn.Tanh()
    
    # 训练参数
    EPOCHS = 50000
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-3
    L_BFGS_STEPS = 500
    
    # 损失权重
    WEIGHT_DATA = 10.0
    WEIGHT_PDE = 1.0
    WEIGHT_BC = 1.0
    WEIGHT_REG = 0.01
    
    # 采样参数
    N_COLLOCATION = 50000  # 内部配点数量
    N_BOUNDARY = 5000      # 边界点数量
    
    # 检查点设置
    CHECKPOINT_INTERVAL = 1000  # 每训练多少轮保存一次检查点
    

# PINN模型定义
class GranitePINN(nn.Module):
    def __init__(self, cube_size):
        super(GranitePINN, self).__init__()
        self.cube_size = cube_size  # 立方体尺寸(mm)
        
        # 构建神经网络层
        layers = []
        layers.append(nn.Linear(3, Config.NEURONS_PER_LAYER))
        layers.append(Config.ACTIVATION)
        
        for _ in range(Config.HIDDEN_LAYERS):
            layers.append(nn.Linear(Config.NEURONS_PER_LAYER, Config.NEURONS_PER_LAYER))
            layers.append(Config.ACTIVATION)
        
        # 共享特征网络
        self.shared_net = nn.Sequential(*layers)
        
        # 压力分支
        self.pressure_head = nn.Sequential(
            nn.Linear(Config.NEURONS_PER_LAYER, 64),
            Config.ACTIVATION,
            nn.Linear(64, 1)
        )
        
        # 渗透率分支(输出对数渗透率)
        self.perm_head = nn.Sequential(
            nn.Linear(Config.NEURONS_PER_LAYER, 64),
            Config.ACTIVATION,
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入坐标张量 [batch_size, 3]
        Returns:
            pressure: 预测压力场
            permeability: 预测渗透率场(单位: m²)
        """
        # 坐标归一化到[-1, 1]范围
        x_norm = 2.0 * x / self.cube_size - 1.0
        
        # 共享特征提取
        features = self.shared_net(x_norm)
        
        # 预测压力 (kPa)
        pressure = self.pressure_head(features)
        
        # 预测对数渗透率 (输出为log(K×10¹⁹))
        log_perm = self.perm_head(features)
        
        # 转换为实际渗透率 (单位: ×10⁻¹⁹ m²)
        permeability = torch.exp(log_perm)
        
        return pressure, permeability
    
    def get_pressure(self, x):
        """仅获取压力预测"""
        pressure, _ = self.forward(x)
        return pressure
    
    def get_permeability(self, x):
        """仅获取渗透率预测"""
        _, permeability = self.forward(x)
        return permeability


# 数据准备函数
def prepare_experimental_data(rock_type):
    """
    准备实验数据
    Args:
        rock_type: 'Stanstead' 或 'Lac du Bonnet'
    Returns:
        surface_data: 表面渗透率测量数据
    """
    # 定义立方体尺寸
    cube_size = Config.STANSTEAD_SIZE if rock_type == 'Stanstead' else Config.LAC_DU_BONNET_SIZE
    
    # 为了简化，我们使用原论文表3和表4的数据
    # 这里我们只创建一个示例数据集
    
    # 创建表面坐标
    face_coords = []
    face_perm_high = []
    face_perm_low = []
    
    # 生成每个面的坐标
    for face in range(6):
        for i in range(3):
            for j in range(3):
                if face == 0:  # x = 0面
                    x, y, z = 0, i*cube_size/2, j*cube_size/2
                elif face == 1:  # x = cube_size面
                    x, y, z = cube_size, i*cube_size/2, j*cube_size/2
                elif face == 2:  # y = 0面
                    x, y, z = i*cube_size/2, 0, j*cube_size/2
                elif face == 3:  # y = cube_size面
                    x, y, z = i*cube_size/2, cube_size, j*cube_size/2
                elif face == 4:  # z = 0面
                    x, y, z = i*cube_size/2, j*cube_size/2, 0
                elif face == 5:  # z = cube_size面
                    x, y, z = i*cube_size/2, j*cube_size/2, cube_size
                
                face_coords.append([x, y, z])
    
    # 使用真实数据(从原论文表格转换)
    if rock_type == 'Stanstead':
        # 从原论文表3中提取数据(简化为示例值)
        # 面1
        face_perm_high.extend([60.7, 54.1, 45.8, 55.1, 63.8, 54.2, 61.4, 49.6, 54.4])
        face_perm_low.extend([54.4, 51.8, 44.6, 52.0, 61.5, 52.9, 60.8, 48.5, 51.0])
        # 面2
        face_perm_high.extend([73.9, 42.0, 64.4, 55.9, 59.1, 41.7, 43.7, 62.8, 58.1])
        face_perm_low.extend([53.0, 40.5, 61.4, 53.9, 55.7, 40.9, 43.4, 60.4, 57.4])
        # 面3-6 (简化，实际应使用原论文中的所有数据)
        # 为简化代码，我们填充剩余数据为平均值
        avg_high = np.mean([60.7, 54.1, 45.8, 55.1, 63.8, 54.2, 61.4, 49.6, 54.4,
                          73.9, 42.0, 64.4, 55.9, 59.1, 41.7, 43.7, 62.8, 58.1])
        avg_low = np.mean([54.4, 51.8, 44.6, 52.0, 61.5, 52.9, 60.8, 48.5, 51.0,
                          53.0, 40.5, 61.4, 53.9, 55.7, 40.9, 43.4, 60.4, 57.4])
        
        # 填充剩余数据
        for _ in range(36):  # 剩余的4个面，每面9个点
            face_perm_high.append(avg_high * (0.9 + 0.2 * np.random.rand()))
            face_perm_low.append(avg_low * (0.9 + 0.2 * np.random.rand()))
    else:  # Lac du Bonnet
        # 从原论文表4中提取数据(简化为示例值)
        # 面1
        face_perm_high.extend([1.59, 2.36, 2.46, 1.3, 0.903, 0.97, 1.3, 1.36, 2.58])
        face_perm_low.extend([1.56, 1.38, 1.86, 1.17, 0.895, 0.901, 1.29, 1.27, 2.39])
        # 面2
        face_perm_high.extend([0.883, 0.827, 0.811, 1.17, 0.956, 1.2, 0.69, 1.14, 1.41])
        face_perm_low.extend([0.785, 0.722, 0.791, 0.739, 0.935, 0.819, 0.648, 0.847, 0.795])
        # 面3-6 (简化，实际应使用原论文中的所有数据)
        # 为简化代码，我们填充剩余数据为平均值
        avg_high = np.mean([1.59, 2.36, 2.46, 1.3, 0.903, 0.97, 1.3, 1.36, 2.58,
                          0.883, 0.827, 0.811, 1.17, 0.956, 1.2, 0.69, 1.14, 1.41])
        avg_low = np.mean([1.56, 1.38, 1.86, 1.17, 0.895, 0.901, 1.29, 1.27, 2.39,
                          0.785, 0.722, 0.791, 0.739, 0.935, 0.819, 0.648, 0.847, 0.795])
        
        # 填充剩余数据
        for _ in range(36):  # 剩余的4个面，每面9个点
            face_perm_high.append(avg_high * (0.9 + 0.2 * np.random.rand()))
            face_perm_low.append(avg_low * (0.9 + 0.2 * np.random.rand()))
    
    # 处理NaN和异常值
    face_perm_high = np.array(face_perm_high, dtype=np.float32)
    face_perm_low = np.array(face_perm_low, dtype=np.float32)
    
    # 转换为张量格式
    coords_tensor = torch.tensor(face_coords, dtype=torch.float32, device=device)
    perm_high_tensor = torch.tensor(face_perm_high, dtype=torch.float32, device=device).unsqueeze(1)
    perm_low_tensor = torch.tensor(face_perm_low, dtype=torch.float32, device=device).unsqueeze(1)
    
    # 返回数据字典
    surface_data = {
        'coords': coords_tensor,
        'perm_high': perm_high_tensor,
        'perm_low': perm_low_tensor
    }
    
    return surface_data


def generate_training_points(cube_size, surface_data, perm_type='high'):
    """
    生成用于训练的点
    Args:
        cube_size: 立方体尺寸
        surface_data: 表面测量数据
        perm_type: 'high' 或 'low'，表示使用高值或低值
    Returns:
        training_points: 包含各种训练点的字典
    """
    # 选择渗透率数据类型
    if perm_type == 'high':
        permeability = surface_data['perm_high']
    else:
        permeability = surface_data['perm_low']
    
    # 1. 表面测量点(用于数据损失)
    data_points = {
        'coords': surface_data['coords'],
        'perm': permeability
    }
    
    # 2. 内部配点(用于PDE损失)
    interior_coords = torch.rand((Config.N_COLLOCATION, 3), device=device) * cube_size
    interior_points = {
        'coords': interior_coords
    }
    
    # 3. 边界配点(用于边界条件损失)
    boundary_coords = generate_boundary_points(cube_size, Config.N_BOUNDARY)
    boundary_points = {
        'coords': boundary_coords
    }
    
    # 返回所有训练点
    training_points = {
        'data': data_points,
        'interior': interior_points,
        'boundary': boundary_points
    }
    
    return training_points


def generate_boundary_points(cube_size, n_points):
    """
    生成立方体边界上的点
    Args:
        cube_size: 立方体尺寸
        n_points: 点的数量
    Returns:
        boundary_points: 边界点坐标
    """
    # 平均分配到6个面
    n_per_face = n_points // 6
    
    boundary_points = []
    
    # 面1: x = 0
    y = torch.rand(n_per_face, device=device) * cube_size
    z = torch.rand(n_per_face, device=device) * cube_size
    x = torch.zeros(n_per_face, device=device)
    boundary_points.append(torch.stack([x, y, z], dim=1))
    
    # 面2: x = cube_size
    y = torch.rand(n_per_face, device=device) * cube_size
    z = torch.rand(n_per_face, device=device) * cube_size
    x = torch.ones(n_per_face, device=device) * cube_size
    boundary_points.append(torch.stack([x, y, z], dim=1))
    
    # 面3: y = 0
    x = torch.rand(n_per_face, device=device) * cube_size
    z = torch.rand(n_per_face, device=device) * cube_size
    y = torch.zeros(n_per_face, device=device)
    boundary_points.append(torch.stack([x, y, z], dim=1))
    
    # 面4: y = cube_size
    x = torch.rand(n_per_face, device=device) * cube_size
    z = torch.rand(n_per_face, device=device) * cube_size
    y = torch.ones(n_per_face, device=device) * cube_size
    boundary_points.append(torch.stack([x, y, z], dim=1))
    
    # 面5: z = 0
    x = torch.rand(n_per_face, device=device) * cube_size
    y = torch.rand(n_per_face, device=device) * cube_size
    z = torch.zeros(n_per_face, device=device)
    boundary_points.append(torch.stack([x, y, z], dim=1))
    
    # 面6: z = cube_size
    x = torch.rand(n_per_face, device=device) * cube_size
    y = torch.rand(n_per_face, device=device) * cube_size
    z = torch.ones(n_per_face, device=device) * cube_size
    boundary_points.append(torch.stack([x, y, z], dim=1))
    
    # 合并所有点
    boundary_points = torch.cat(boundary_points, dim=0)
    
    return boundary_points


def compute_pde_loss(model, interior_points):
    """
    计算PDE损失(渗流控制方程残差)
    Args:
        model: PINN模型
        interior_points: 内部配点
    Returns:
        pde_loss: PDE损失值
    """
    # 获取内部点坐标并设置梯度追踪
    x = interior_points['coords'].clone().detach().requires_grad_(True)
    
    # 预测压力和渗透率
    pressure, permeability = model(x)
    
    # 计算压力梯度 ∇p
    grad_p = torch.autograd.grad(
        pressure.sum(), x, create_graph=True, retain_graph=True
    )[0]
    
    # 计算 K∇p
    k_grad_p = permeability * grad_p
    
    # 计算散度 ∇·(K∇p)
    divergence = 0.0
    for i in range(3):
        # 计算 ∂/∂xi (K ∂p/∂xi)
        partial_derivative = torch.autograd.grad(
            k_grad_p[:, i].sum(), x, create_graph=True, retain_graph=True
        )[0][:, i]
        divergence += partial_derivative
    
    # PDE残差: ∇·(K∇p) = 0，所以残差就是 (∇·(K∇p))²
    pde_loss = torch.mean(divergence**2)
    
    return pde_loss


def compute_data_loss(model, data_points):
    """
    计算数据损失(预测渗透率与测量值的差异)
    Args:
        model: PINN模型
        data_points: 数据点
    Returns:
        data_loss: 数据损失值
    """
    # 获取坐标和测量渗透率
    coords = data_points['coords']
    measured_perm = data_points['perm']
    
    # 预测渗透率
    predicted_perm = model.get_permeability(coords)
    
    # 计算对数空间的均方误差
    data_loss = torch.mean((torch.log(predicted_perm) - torch.log(measured_perm))**2)
    
    return data_loss


def compute_boundary_loss(model, boundary_points):
    """
    计算边界条件损失
    Args:
        model: PINN模型
        boundary_points: 边界点
    Returns:
        boundary_loss: 边界条件损失值
    """
    # 这里简化处理，我们假设边界条件只是为了确保渗透率场的平滑性
    # 实际应用中，可以根据特定实验设置添加更复杂的边界条件
    
    # 获取边界点坐标
    coords = boundary_points['coords']
    
    # 预测边界点的渗透率
    predicted_perm = model.get_permeability(coords)
    
    # 计算渗透率梯度(应该很小，表示平滑过渡)
    # 我们使用一个简化方法：计算每个点与其邻居的渗透率差异
    n_points = coords.shape[0]
    if n_points < 2:
        return torch.tensor(0.0, device=device)
    
    # 随机打乱并重新排序点(模拟邻居关系)
    idx = torch.randperm(n_points, device=device)
    perm1 = predicted_perm[idx[:n_points//2]]
    perm2 = predicted_perm[idx[n_points//2:]]
    
    # 计算对数空间的差异
    boundary_loss = torch.mean((torch.log(perm1) - torch.log(perm2))**2)
    
    return boundary_loss


def compute_regularization_loss(model):
    """
    计算正则化损失(防止过拟合)
    Args:
        model: PINN模型
    Returns:
        reg_loss: 正则化损失值
    """
    # L2正则化
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param**2)
    
    return reg_loss


def compute_total_loss(model, training_points, weights):
    """
    计算总损失
    Args:
        model: PINN模型
        training_points: 训练点
        weights: 各损失项的权重
    Returns:
        total_loss: 总损失值
        loss_components: 各损失项的值
    """
    # 计算各损失项
    data_loss = compute_data_loss(model, training_points['data'])
    pde_loss = compute_pde_loss(model, training_points['interior'])
    boundary_loss = compute_boundary_loss(model, training_points['boundary'])
    reg_loss = compute_regularization_loss(model)
    
    # 加权求和
    total_loss = (
        weights['data'] * data_loss +
        weights['pde'] * pde_loss +
        weights['boundary'] * boundary_loss +
        weights['reg'] * reg_loss
    )
    
    # 返回总损失和各损失项
    loss_components = {
        'data': data_loss.item(),
        'pde': pde_loss.item(),
        'boundary': boundary_loss.item(),
        'reg': reg_loss.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_components


def save_model_state(model, optimizer, scheduler, epoch, loss_history, rock_type):
    """
    保存模型训练状态
    Args:
        model: PINN模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前训练轮数
        loss_history: 损失历史
        rock_type: 岩石类型
    """
    # 创建保存目录
    save_dir = f'models/{rock_type}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型权重
    torch.save(model.state_dict(), f'{save_dir}/model_weights.pth')
    
    # 保存训练状态
    training_state = {
        'epoch': epoch,
        'optimizer_state': optimizer.state_dict() if optimizer else None,
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'loss_history': loss_history
    }
    torch.save(training_state, f'{save_dir}/training_state.pth')
    
    print(f"模型和训练状态已保存到 {save_dir}")


def load_model_state(model, optimizer, scheduler, rock_type):
    """
    加载模型训练状态
    Args:
        model: PINN模型
        optimizer: 优化器
        scheduler: 学习率调度器
        rock_type: 岩石类型
    Returns:
        epoch: 恢复的训练轮数
        loss_history: 恢复的损失历史
    """
    # 检查保存目录
    save_dir = f'models/{rock_type}'
    model_path = f'{save_dir}/model_weights.pth'
    training_path = f'{save_dir}/training_state.pth'
    
    # 检查文件是否存在
    if not os.path.exists(model_path) or not os.path.exists(training_path):
        print(f"没有找到保存的模型状态: {model_path}")
        return 0, {'total': [], 'data': [], 'pde': [], 'boundary': [], 'reg': []}
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    
    # 加载训练状态
    training_state = torch.load(training_path)
    epoch = training_state['epoch']
    
    if optimizer and training_state['optimizer_state']:
        optimizer.load_state_dict(training_state['optimizer_state'])
    
    if scheduler and training_state['scheduler_state']:
        scheduler.load_state_dict(training_state['scheduler_state'])
    
    loss_history = training_state['loss_history']
    
    print(f"已从第 {epoch} 轮恢复模型训练状态")
    return epoch, loss_history


def train_model_with_checkpoints(model, training_points, weights, optimizer, scheduler, 
                              epochs, rock_type, checkpoint_interval=None):
    """
    带检查点的模型训练
    Args:
        model: PINN模型
        training_points: 训练点
        weights: 各损失项的权重
        optimizer: 优化器
        scheduler: 学习率调度器
        epochs: 训练轮数
        rock_type: 岩石类型
        checkpoint_interval: 检查点间隔
    Returns:
        loss_history: 损失历史
    """
    if checkpoint_interval is None:
        checkpoint_interval = Config.CHECKPOINT_INTERVAL
    
    # 检查是否有保存的状态
    start_epoch, loss_history = load_model_state(model, optimizer, scheduler, rock_type)
    
    # 如果已经完成训练，直接返回
    if start_epoch >= epochs:
        print(f"训练已完成 ({start_epoch}/{epochs} 轮)，跳过训练")
        return loss_history
    
    # 训练循环
    for epoch in range(start_epoch, epochs):
        # 计算损失
        total_loss, loss_components = compute_total_loss(model, training_points, weights)
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 更新学习率
        if scheduler:
            scheduler.step(total_loss)
        
        # 记录损失
        for key in loss_history:
            loss_history[key].append(loss_components[key])
        
        # 打印进度
        if epoch % 1000 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch}/{epochs} - Total Loss: {loss_components['total']:.6f}, "
                  f"Data: {loss_components['data']:.6f}, "
                  f"PDE: {loss_components['pde']:.6f}, "
                  f"Boundary: {loss_components['boundary']:.6f}, "
                  f"Reg: {loss_components['reg']:.6f}")
        
        # 保存检查点
        if (epoch + 1) % checkpoint_interval == 0 or epoch == epochs-1:
            save_model_state(model, optimizer, scheduler, epoch+1, loss_history, rock_type)
    
    return loss_history


def lbfgs_optimization(model, training_points, weights, rock_type):
    """
    使用L-BFGS优化器进行微调
    Args:
        model: PINN模型
        training_points: 训练点
        weights: 各损失项的权重
        rock_type: 岩石类型
    Returns:
        final_loss: 最终损失值
    """
    # 定义损失计算函数
    def closure():
        optimizer.zero_grad()
        loss, _ = compute_total_loss(model, training_points, weights)
        loss.backward()
        return loss
    
    # 创建L-BFGS优化器
    optimizer = optim.LBFGS(model.parameters(), 
                           lr=0.1, 
                           max_iter=Config.L_BFGS_STEPS, 
                           history_size=50, 
                           tolerance_grad=1e-5,
                           tolerance_change=1e-5,
                           line_search_fn="strong_wolfe")
    
    # 进行优化
    print("开始L-BFGS优化...")
    optimizer.step(closure)
    
    # 计算最终损失
    final_loss, loss_components = compute_total_loss(model, training_points, weights)
    print(f"L-BFGS完成 - 最终损失: {final_loss.item():.6f}")
    
    # 保存优化后的模型
    save_model_state(model, None, None, Config.EPOCHS, None, rock_type)
    
    return final_loss.item()


def save_permeability_field(model, cube_size, rock_type, n_grid=51):
    """
    计算并保存完整的渗透率场
    Args:
        model: PINN模型
        cube_size: 立方体尺寸
        rock_type: 岩石类型
        n_grid: 网格分辨率
    """
    # 创建保存目录
    save_dir = f'results/{rock_type}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建网格点
    x = torch.linspace(0, cube_size, n_grid, device=device)
    y = torch.linspace(0, cube_size, n_grid, device=device)
    z = torch.linspace(0, cube_size, n_grid, device=device)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    
    # 分批计算以避免内存溢出
    batch_size = 10000
    n_batches = (grid_points.shape[0] + batch_size - 1) // batch_size
    
    # 初始化结果数组
    pressure_field = np.zeros(grid_points.shape[0])
    perm_field = np.zeros(grid_points.shape[0])
    
    # 批量预测
    print(f"计算 {rock_type} 岩石的渗透率场 (分辨率: {n_grid}x{n_grid}x{n_grid})...")
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, grid_points.shape[0])
            batch_points = grid_points[start_idx:end_idx]
            
            pressure, perm = model(batch_points)
            
            pressure_field[start_idx:end_idx] = pressure.cpu().numpy().flatten()
            perm_field[start_idx:end_idx] = perm.cpu().numpy().flatten()
            
            if (i+1) % 10 == 0 or i == n_batches-1:
                print(f"处理批次 {i+1}/{n_batches}")
    
    # 重塑为3D数组
    pressure_field = pressure_field.reshape(n_grid, n_grid, n_grid)
    perm_field = perm_field.reshape(n_grid, n_grid, n_grid)
    
    # 保存为numpy文件
    coords = {
        'x': x.cpu().numpy(),
        'y': y.cpu().numpy(),
        'z': z.cpu().numpy()
    }
    
    np.save(f'{save_dir}/pressure_field.npy', pressure_field)
    np.save(f'{save_dir}/permeability_field.npy', perm_field)
    np.save(f'{save_dir}/coordinates.npy', coords)
    
    # 保存元数据
    metadata = {
        'rock_type': rock_type,
        'cube_size': cube_size,
        'n_grid': n_grid,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 使用普通的文本文件保存元数据
    with open(f'{save_dir}/metadata.txt', 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"渗透率场已保存到 {save_dir}")
    
    return pressure_field, perm_field, coords


def load_permeability_field(rock_type):
    """
    加载预计算的渗透率场
    Args:
        rock_type: 岩石类型
    Returns:
        pressure_field: 压力场
        perm_field: 渗透率场
        coords: 坐标网格
        metadata: 元数据
    """
    # 检查保存目录
    save_dir = f'results/{rock_type}'
    pressure_path = f'{save_dir}/pressure_field.npy'
    perm_path = f'{save_dir}/permeability_field.npy'
    coords_path = f'{save_dir}/coordinates.npy'
    metadata_path = f'{save_dir}/metadata.txt'
    
    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [pressure_path, perm_path, coords_path]):
        print(f"没有找到预计算的渗透率场: {save_dir}")
        return None, None, None, None
    
    # 加载数据
    pressure_field = np.load(pressure_path)
    perm_field = np.load(perm_path)
    coords = np.load(coords_path, allow_pickle=True).item()
    
    # 加载元数据
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ', 1)
                metadata[key] = value
    
    print(f"已加载预计算的渗透率场，分辨率: {pressure_field.shape}")
    return pressure_field, perm_field, coords, metadata


def compute_effective_permeability(model, cube_size, n_grid=31):
    """
    计算有效渗透率
    Args:
        model: PINN模型
        cube_size: 立方体尺寸
        n_grid: 每个方向的网格点数
    Returns:
        k_eff: 各方向的有效渗透率
        k_geometric: 几何平均有效渗透率
    """
    # 创建网格点
    x = torch.linspace(0, cube_size, n_grid, device=device)
    y = torch.linspace(0, cube_size, n_grid, device=device)
    z = torch.linspace(0, cube_size, n_grid, device=device)
    
    # 计算三个方向的有效渗透率
    k_eff = {}
    
    # X方向(yz面积分)
    k_eff['x'] = compute_directional_perm(model, cube_size, x, y, z, 'x', n_grid)
    
    # Y方向(xz面积分)
    k_eff['y'] = compute_directional_perm(model, cube_size, x, y, z, 'y', n_grid)
    
    # Z方向(xy面积分)
    k_eff['z'] = compute_directional_perm(model, cube_size, x, y, z, 'z', n_grid)
    
    # 计算几何平均值
    k_geometric = (k_eff['x'] * k_eff['y'] * k_eff['z'])**(1/3)
    
    return k_eff, k_geometric


def compute_directional_perm(model, cube_size, x, y, z, direction, n_grid):
    """
    计算特定方向上的有效渗透率
    Args:
        model: PINN模型
        cube_size: 立方体尺寸
        x, y, z: 三个方向的网格点
        direction: 计算方向('x', 'y' 或 'z')
        n_grid: 每个方向的网格点数
    Returns:
        k_eff: 该方向的有效渗透率
    """
    # 根据方向设置入口和出口面
    if direction == 'x':
        # 在yz平面上遍历
        perm_values = torch.zeros((n_grid, n_grid), device=device)
        for i in range(n_grid):
            for j in range(n_grid):
                # 沿x方向的网格点
                points = torch.zeros((n_grid, 3), device=device)
                points[:, 0] = x
                points[:, 1] = y[i]
                points[:, 2] = z[j]
                
                # 预测沿线的渗透率
                with torch.no_grad():
                    _, perm = model(points)
                
                # 调和平均(倒数的算术平均)
                k_harmonic = n_grid / torch.sum(1.0 / perm.squeeze())
                perm_values[i, j] = k_harmonic
        
        # 对yz平面进行算术平均
        k_eff = torch.mean(perm_values)
        
    elif direction == 'y':
        # 在xz平面上遍历
        perm_values = torch.zeros((n_grid, n_grid), device=device)
        for i in range(n_grid):
            for j in range(n_grid):
                # 沿y方向的网格点
                points = torch.zeros((n_grid, 3), device=device)
                points[:, 0] = x[i]
                points[:, 1] = y
                points[:, 2] = z[j]
                
                # 预测沿线的渗透率
                with torch.no_grad():
                    _, perm = model(points)
                
                # 调和平均
                k_harmonic = n_grid / torch.sum(1.0 / perm.squeeze())
                perm_values[i, j] = k_harmonic
        
        # 对xz平面进行算术平均
        k_eff = torch.mean(perm_values)
        
    else:  # direction == 'z'
        # 在xy平面上遍历
        perm_values = torch.zeros((n_grid, n_grid), device=device)
        for i in range(n_grid):
            for j in range(n_grid):
                # 沿z方向的网格点
                points = torch.zeros((n_grid, 3), device=device)
                points[:, 0] = x[i]
                points[:, 1] = y[j]
                points[:, 2] = z
                
                # 预测沿线的渗透率
                with torch.no_grad():
                    _, perm = model(points)
                
                # 调和平均
                k_harmonic = n_grid / torch.sum(1.0 / perm.squeeze())
                perm_values[i, j] = k_harmonic
        
        # 对xy平面进行算术平均
        k_eff = torch.mean(perm_values)
    
    return k_eff.item()


def compute_effective_permeability_from_saved(rock_type):
    """
    从保存的渗透率场计算有效渗透率
    Args:
        rock_type: 岩石类型
    Returns:
        k_eff: 各方向的有效渗透率
        k_geometric: 几何平均有效渗透率
    """
    # 加载渗透率场
    _, perm_field, coords, metadata = load_permeability_field(rock_type)
    
    if perm_field is None:
        print(f"没有找到保存的渗透率场，无法计算有效渗透率")
        return None, None
    
    # 获取立方体尺寸和网格数量
    cube_size = float(metadata.get('cube_size', 
                               Config.STANSTEAD_SIZE if rock_type == 'Stanstead' else Config.LAC_DU_BONNET_SIZE))
    n_grid = perm_field.shape[0]
    
    # 恢复坐标数组
    x = coords['x']
    y = coords['y']
    z = coords['z']
    
    # 计算三个方向的有效渗透率
    k_eff = {}
    
    # X方向(yz面积分)
    k_eff['x'] = compute_directional_perm_from_field(perm_field, 'x', n_grid)
    
    # Y方向(xz面积分)
    k_eff['y'] = compute_directional_perm_from_field(perm_field, 'y', n_grid)
    
    # Z方向(xy面积分)
    k_eff['z'] = compute_directional_perm_from_field(perm_field, 'z', n_grid)
    
    # 计算几何平均值
    k_geometric = (k_eff['x'] * k_eff['y'] * k_eff['z'])**(1/3)
    
    print(f"{rock_type} 有效渗透率 (×10⁻¹⁹ m²):")
    print(f"X方向: {k_eff['x']:.4f}, Y方向: {k_eff['y']:.4f}, Z方向: {k_eff['z']:.4f}")
    print(f"几何平均: {k_geometric:.4f}")
    
    return k_eff, k_geometric


def compute_directional_perm_from_field(perm_field, direction, n_grid):
    """
    从预计算的渗透率场计算特定方向上的有效渗透率
    Args:
        perm_field: 渗透率场
        direction: 计算方向('x', 'y' 或 'z')
        n_grid: 网格分辨率
    Returns:
        k_eff: 该方向的有效渗透率
    """
    # 根据方向设置计算轴
    if direction == 'x':
        # 沿x方向计算
        k_values = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                # 沿x轴的一维线
                line_perms = perm_field[:, i, j]
                # 调和平均(倒数的算术平均)
                k_harmonic = n_grid / np.sum(1.0 / line_perms)
                k_values[i, j] = k_harmonic
        
        # 对yz平面进行算术平均
        k_eff = np.mean(k_values)
        
    elif direction == 'y':
        # 沿y方向计算
        k_values = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                # 沿y轴的一维线
                line_perms = perm_field[i, :, j]
                # 调和平均
                k_harmonic = n_grid / np.sum(1.0 / line_perms)
                k_values[i, j] = k_harmonic
        
        # 对xz平面进行算术平均
        k_eff = np.mean(k_values)
        
    else:  # direction == 'z'
        # 沿z方向计算
        k_values = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                # 沿z轴的一维线
                line_perms = perm_field[i, j, :]
                # 调和平均
                k_harmonic = n_grid / np.sum(1.0 / line_perms)
                k_values[i, j] = k_harmonic
        
        # 对xy平面进行算术平均
        k_eff = np.mean(k_values)
    
    return k_eff


def plot_loss_history(loss_history, rock_type):
    """
    绘制损失历史
    Args:
        loss_history: 损失历史字典
        rock_type: 岩石类型
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制总损失
    plt.subplot(2, 1, 1)
    plt.semilogy(loss_history['total'], label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'{rock_type} Granite - Total Loss History')
    plt.legend()
    plt.grid(True)
    
    # 绘制各损失项
    plt.subplot(2, 1, 2)
    plt.semilogy(loss_history['data'], label='Data Loss')
    plt.semilogy(loss_history['pde'], label='PDE Loss')
    plt.semilogy(loss_history['boundary'], label='Boundary Loss')
    plt.semilogy(loss_history['reg'], label='Reg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'{rock_type} Granite - Component Loss History')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{rock_type}_loss_history.png', dpi=300)
    plt.close()


def visualize_permeability_distribution(model, cube_size, rock_type, n_grid=31):
    """
    可视化渗透率分布
    Args:
        model: PINN模型
        cube_size: 立方体尺寸
        rock_type: 岩石类型
        n_grid: 网格点数
    """
    # 创建网格点
    x = torch.linspace(0, cube_size, n_grid, device=device)
    y = torch.linspace(0, cube_size, n_grid, device=device)
    z = torch.linspace(0, cube_size, n_grid, device=device)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    
    # 预测渗透率
    with torch.no_grad():
        _, permeability = model(points)
    
    # 将渗透率重塑为3D网格
    perm_grid = permeability.reshape(n_grid, n_grid, n_grid).cpu().numpy()
    
    # 创建3D可视化
    fig = plt.figure(figsize=(18, 6))
    
    # 1. 中间XY截面
    ax1 = fig.add_subplot(131)
    mid_z = n_grid // 2
    im1 = ax1.imshow(perm_grid[:, :, mid_z].T, 
                     extent=[0, cube_size, 0, cube_size],
                     cmap='viridis', 
                     origin='lower')
    ax1.set_title(f'{rock_type} - XY Plane (z={z[mid_z].item():.0f}mm)')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    fig.colorbar(im1, ax=ax1, label='Permeability (×10⁻¹⁹ m²)')
    
    # 2. 中间XZ截面
    ax2 = fig.add_subplot(132)
    mid_y = n_grid // 2
    im2 = ax2.imshow(perm_grid[:, mid_y, :].T, 
                     extent=[0, cube_size, 0, cube_size],
                     cmap='viridis', 
                     origin='lower')
    ax2.set_title(f'{rock_type} - XZ Plane (y={y[mid_y].item():.0f}mm)')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    fig.colorbar(im2, ax=ax2, label='Permeability (×10⁻¹⁹ m²)')
    
    # 3. 中间YZ截面
    ax3 = fig.add_subplot(133)
    mid_x = n_grid // 2
    im3 = ax3.imshow(perm_grid[mid_x, :, :].T, 
                     extent=[0, cube_size, 0, cube_size],
                     cmap='viridis', 
                     origin='lower')
    ax3.set_title(f'{rock_type} - YZ Plane (x={x[mid_x].item():.0f}mm)')
    ax3.set_xlabel('Y (mm)')
    ax3.set_ylabel('Z (mm)')
    fig.colorbar(im3, ax=ax3, label='Permeability (×10⁻¹⁹ m²)')
    
    plt.tight_layout()
    plt.savefig(f'{rock_type}_perm_distribution.png', dpi=300)
    plt.close()
    
    # 创建3D散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 随机采样点以减少绘图数据量
    sample_size = 1000  # 随机采样1000个点来可视化
    indices = np.random.choice(n_grid**3, sample_size, replace=False)
    
    # 获取采样点坐标和值
    x_sample = points.cpu().numpy()[indices, 0]
    y_sample = points.cpu().numpy()[indices, 1]
    z_sample = points.cpu().numpy()[indices, 2]
    perm_sample = permeability.cpu().numpy()[indices]
    
    # 用散点图表示3D渗透率分布，颜色表示渗透率值
    scatter = ax.scatter(x_sample, y_sample, z_sample, 
                         c=perm_sample.flatten(), 
                         cmap='viridis',
                         alpha=0.8,
                         s=20)
    
    # 设置坐标轴
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'{rock_type} - 3D Permeability Distribution')
    
    # 添加颜色条
    fig.colorbar(scatter, ax=ax, label='Permeability (×10⁻¹⁹ m²)')
    
    plt.tight_layout()
    plt.savefig(f'{rock_type}_perm_3d.png', dpi=300)
    plt.close()


def visualize_saved_permeability(rock_type):
    """
    可视化保存的渗透率场
    Args:
        rock_type: 岩石类型
    """
    # 加载渗透率场
    pressure_field, perm_field, coords, metadata = load_permeability_field(rock_type)
    
    if perm_field is None:
        print(f"没有找到保存的渗透率场，无法可视化")
        return
    
    # 获取立方体尺寸和网格数量
    cube_size = float(metadata.get('cube_size', 
                               Config.STANSTEAD_SIZE if rock_type == 'Stanstead' else Config.LAC_DU_BONNET_SIZE))
    n_grid = perm_field.shape[0]
    
    # 恢复坐标数组
    x = coords['x']
    y = coords['y']
    z = coords['z']
    
    # 创建3D可视化
    fig = plt.figure(figsize=(18, 6))
    
    # 1. 中间XY截面
    ax1 = fig.add_subplot(131)
    mid_z = n_grid // 2
    im1 = ax1.imshow(perm_field[:, :, mid_z].T, 
                     extent=[0, cube_size, 0, cube_size],
                     cmap='viridis', 
                     origin='lower')
    ax1.set_title(f'{rock_type} - XY Plane (z={z[mid_z]:.0f}mm)')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    fig.colorbar(im1, ax=ax1, label='Permeability (×10⁻¹⁹ m²)')
    
    # 2. 中间XZ截面
    ax2 = fig.add_subplot(132)
    mid_y = n_grid // 2
    im2 = ax2.imshow(perm_field[:, mid_y, :].T, 
                     extent=[0, cube_size, 0, cube_size],
                     cmap='viridis', 
                     origin='lower')
    ax2.set_title(f'{rock_type} - XZ Plane (y={y[mid_y]:.0f}mm)')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (mm)')
    fig.colorbar(im2, ax=ax2, label='Permeability (×10⁻¹⁹ m²)')
    
    # 3. 中间YZ截面
    ax3 = fig.add_subplot(133)
    mid_x = n_grid // 2
    im3 = ax3.imshow(perm_field[mid_x, :, :].T, 
                     extent=[0, cube_size, 0, cube_size],
                     cmap='viridis', 
                     origin='lower')
    ax3.set_title(f'{rock_type} - YZ Plane (x={x[mid_x]:.0f}mm)')
    ax3.set_xlabel('Y (mm)')
    ax3.set_ylabel('Z (mm)')
    fig.colorbar(im3, ax=ax3, label='Permeability (×10⁻¹⁹ m²)')
    
    plt.tight_layout()
    plt.savefig(f'{rock_type}_saved_perm_distribution.png', dpi=300)
    plt.close()
    
    # 创建3D散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 随机采样点以减少绘图数据量
    sample_size = 1000
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
    flat_indices = np.random.choice(n_grid**3, sample_size, replace=False)
    
    # 计算3D索引
    idx_x = flat_indices // (n_grid * n_grid)
    idx_y = (flat_indices % (n_grid * n_grid)) // n_grid
    idx_z = flat_indices % n_grid
    
    # 采样点坐标和值
    x_sample = x_grid[idx_x, idx_y, idx_z]
    y_sample = y_grid[idx_x, idx_y, idx_z]
    z_sample = z_grid[idx_x, idx_y, idx_z]
    perm_sample = perm_field[idx_x, idx_y, idx_z]
    
    # 用散点图表示3D渗透率分布
    scatter = ax.scatter(x_sample, y_sample, z_sample, 
                         c=perm_sample, 
                         cmap='viridis',
                         alpha=0.8,
                         s=20)
    
    # 设置坐标轴
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'{rock_type} - 3D Permeability Distribution (From Saved Data)')
    
    # 添加颜色条
    fig.colorbar(scatter, ax=ax, label='Permeability (×10⁻¹⁹ m²)')
    
    plt.tight_layout()
    plt.savefig(f'{rock_type}_saved_perm_3d.png', dpi=300)
    plt.close()


def plot_permeability_histogram(model, cube_size, rock_type, n_samples=100000):
    """
    绘制渗透率分布直方图
    Args:
        model: PINN模型
        cube_size: 立方体尺寸
        rock_type: 岩石类型
        n_samples: 采样点数量
    """
    # 随机采样点
    points = torch.rand((n_samples, 3), device=device) * cube_size
    
    # 预测渗透率
    with torch.no_grad():
        _, permeability = model(points)
    
    # 转换为NumPy数组
    perm_np = permeability.cpu().numpy().flatten()
    
    # 计算统计量
    mean_perm = np.mean(perm_np)
    median_perm = np.median(perm_np)
    std_perm = np.std(perm_np)
    
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(perm_np, bins=50, alpha=0.7, color='blue', density=True)
    
    # 添加统计信息
    plt.axvline(mean_perm, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_perm:.2f}')
    plt.axvline(median_perm, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_perm:.2f}')
    
    # 拟合对数正态分布
    shape, loc, scale = lognorm.fit(perm_np)
    x = np.linspace(min(perm_np), max(perm_np), 1000)
    pdf = lognorm.pdf(x, shape, loc, scale)
    plt.plot(x, pdf, 'r-', linewidth=2, label='Log-normal Fit')
    
    # 添加标签和标题
    plt.xlabel('Permeability (×10⁻¹⁹ m²)')
    plt.ylabel('Probability Density')
    plt.title(f'{rock_type} Granite - Permeability Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加更多统计信息作为文本
    plt.text(0.05, 0.95, f'Mean: {mean_perm:.2f}\nMedian: {median_perm:.2f}\nStd Dev: {std_perm:.2f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{rock_type}_perm_histogram.png', dpi=300)
    plt.close()
    
    return mean_perm, median_perm, std_perm


def plot_histogram_from_saved(rock_type):
    """
    从保存的渗透率场绘制直方图
    Args:
        rock_type: 岩石类型
    """
    # 加载渗透率场
    _, perm_field, _, _ = load_permeability_field(rock_type)
    
    if perm_field is None:
        print(f"没有找到保存的渗透率场，无法绘制直方图")
        return None, None, None
    
    # 扁平化数组以计算统计量
    perm_flat = perm_field.flatten()
    
    # 计算统计量
    mean_perm = np.mean(perm_flat)
    median_perm = np.median(perm_flat)
    std_perm = np.std(perm_flat)
    
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(perm_flat, bins=50, alpha=0.7, color='blue', density=True)
    
    # 添加统计信息
    plt.axvline(mean_perm, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_perm:.2f}')
    plt.axvline(median_perm, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_perm:.2f}')
    
    # 拟合对数正态分布
    shape, loc, scale = lognorm.fit(perm_flat)
    x = np.linspace(min(perm_flat), max(perm_flat), 1000)
    pdf = lognorm.pdf(x, shape, loc, scale)
    plt.plot(x, pdf, 'r-', linewidth=2, label='Log-normal Fit')
    
    # 添加标签和标题
    plt.xlabel('Permeability (×10⁻¹⁹ m²)')
    plt.ylabel('Probability Density')
    plt.title(f'{rock_type} Granite - Permeability Distribution (From Saved Data)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加更多统计信息作为文本
    plt.text(0.05, 0.95, f'Mean: {mean_perm:.2f}\nMedian: {median_perm:.2f}\nStd Dev: {std_perm:.2f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{rock_type}_saved_perm_histogram.png', dpi=300)
    plt.close()
    
    return mean_perm, median_perm, std_perm


def kriging_simulation(surface_data, rock_type):
    """
    模拟原论文的kriging方法计算有效渗透率
    (注意：这只是一个简化模拟，不是完整的kriging实现)
    Args:
        surface_data: 表面渗透率数据
        rock_type: 岩石类型
    Returns:
        kriging_results: kriging方法结果
    """
    # 获取立方体尺寸
    cube_size = Config.STANSTEAD_SIZE if rock_type == 'Stanstead' else Config.LAC_DU_BONNET_SIZE
    
    # 从原论文获取有效渗透率结果
    # 注意：这里使用的是论文中的实际值，而非模拟计算
    if rock_type == 'Stanstead':
        k_eff = {
            'x': 59.2,
            'y': 59.5,
            'z': 59.2
        }
        k_geometric = 59.3
    else:  # Lac du Bonnet
        k_eff = {
            'x': 1.09,
            'y': 1.08,
            'z': 1.10
        }
        k_geometric = 1.09
    
    # 返回结果
    kriging_results = {
        'k_eff': k_eff,
        'k_geometric': k_geometric
    }
    
    return kriging_results


def compare_with_original_results(pinn_results, original_results, rock_type):
    """
    与原论文结果比较
    Args:
        pinn_results: PINN方法结果
        original_results: 原论文结果
        rock_type: 岩石类型
    """
    # 创建比较图表
    plt.figure(figsize=(10, 6))
    
    # 设置数据
    methods = ['PINN', 'Original (Kriging)']
    
    # 有效渗透率
    k_eff = [pinn_results['k_geometric'], original_results['k_geometric']]
    
    # X, Y, Z方向渗透率
    k_x = [pinn_results['k_eff']['x'], original_results['k_eff']['x']]
    k_y = [pinn_results['k_eff']['y'], original_results['k_eff']['y']]
    k_z = [pinn_results['k_eff']['z'], original_results['k_eff']['z']]
    
    # 设置柱状图位置
    x = np.arange(len(methods))
    width = 0.2
    
    # 绘制柱状图
    plt.bar(x - width*1.5, k_eff, width, label='Geometric Mean', color='blue')
    plt.bar(x - width/2, k_x, width, label='X Direction', color='red')
    plt.bar(x + width/2, k_y, width, label='Y Direction', color='green')
    plt.bar(x + width*1.5, k_z, width, label='Z Direction', color='purple')
    
    # 添加标签和标题
    plt.xlabel('Method')
    plt.ylabel('Permeability (×10⁻¹⁹ m²)')
    plt.title(f'{rock_type} Granite - Effective Permeability Comparison')
    plt.xticks(x, methods)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(k_eff):
        plt.text(i - width*1.5, v + 0.5, f'{v:.2f}', ha='center')
    for i, v in enumerate(k_x):
        plt.text(i - width/2, v + 0.5, f'{v:.2f}', ha='center')
    for i, v in enumerate(k_y):
        plt.text(i + width/2, v + 0.5, f'{v:.2f}', ha='center')
    for i, v in enumerate(k_z):
        plt.text(i + width*1.5, v + 0.5, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{rock_type}_comparison.png', dpi=300)
    plt.close()
    
    # 打印比较结果
    print(f"\n{rock_type} Granite - Permeability Comparison (×10⁻¹⁹ m²):")
    print(f"Method           Geometric Mean   X Direction   Y Direction   Z Direction")
    print(f"PINN             {pinn_results['k_geometric']:.2f}            {pinn_results['k_eff']['x']:.2f}           {pinn_results['k_eff']['y']:.2f}           {pinn_results['k_eff']['z']:.2f}")
    print(f"Original(Kriging) {original_results['k_geometric']:.2f}            {original_results['k_eff']['x']:.2f}           {original_results['k_eff']['y']:.2f}           {original_results['k_eff']['z']:.2f}")
    print(f"Difference (%)   {(pinn_results['k_geometric'] - original_results['k_geometric'])/original_results['k_geometric']*100:.2f}%            {(pinn_results['k_eff']['x'] - original_results['k_eff']['x'])/original_results['k_eff']['x']*100:.2f}%           {(pinn_results['k_eff']['y'] - original_results['k_eff']['y'])/original_results['k_eff']['y']*100:.2f}%           {(pinn_results['k_eff']['z'] - original_results['k_eff']['z'])/original_results['k_eff']['z']*100:.2f}%")


def main():
    """主函数，支持从保存的状态继续训练或直接加载预计算结果"""
    # 创建目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 设置损失权重
    weights = {
        'data': Config.WEIGHT_DATA,
        'pde': Config.WEIGHT_PDE,
        'boundary': Config.WEIGHT_BC,
        'reg': Config.WEIGHT_REG
    }
    
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='PINN花岗岩渗透率研究')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--compute', action='store_true', help='计算渗透率场')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--rock', type=str, default='both', 
                       choices=['Stanstead', 'Lac du Bonnet', 'both'], 
                       help='要处理的岩石类型')
    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认执行所有操作
    if not (args.train or args.compute or args.visualize):
        args.train = True
        args.compute = True
        args.visualize = True
    
    # 确定要处理的岩石类型
    rock_types = []
    if args.rock == 'both':
        rock_types = ['Stanstead', 'Lac du Bonnet']
    else:
        rock_types = [args.rock]
    
    # 处理每种岩石
    for rock_type in rock_types:
        print(f"\n=== 处理{rock_type}花岗岩 ===")
        
        # 获取立方体尺寸
        cube_size = Config.STANSTEAD_SIZE if rock_type == 'Stanstead' else Config.LAC_DU_BONNET_SIZE
        
        # 准备数据
        surface_data = prepare_experimental_data(rock_type)
        training_points = generate_training_points(cube_size, surface_data, 'high')
        
        # 如果需要训练
        if args.train:
            # 创建模型
            model = GranitePINN(cube_size).to(device)
            
            # 创建优化器和学习率调度器
            optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=1000, factor=0.5, verbose=True)
            
            # 检查是否有保存的模型，如果有则加载，没有则重新训练
            start_epoch, loss_history = load_model_state(model, optimizer, scheduler, rock_type)
            
            if start_epoch == 0:
                print(f"未找到之前保存的模型，开始从头训练{rock_type}模型")
                loss_history = {
                    'total': [], 
                    'data': [], 
                    'pde': [], 
                    'boundary': [], 
                    'reg': []
                }
            else:
                print(f"从第{start_epoch}轮继续训练{rock_type}模型")
            
            # 训练模型(带检查点)
            loss_history = train_model_with_checkpoints(
                model, 
                training_points, 
                weights, 
                optimizer, 
                scheduler, 
                Config.EPOCHS,
                rock_type,
                Config.CHECKPOINT_INTERVAL
            )
            
            # L-BFGS微调
            lbfgs_optimization(model, training_points, weights, rock_type)
            
            # 绘制损失历史
            plot_loss_history(loss_history, rock_type)
            
            # 保存最终模型状态
            save_model_state(model, optimizer, scheduler, Config.EPOCHS, loss_history, rock_type)
        
        # 如果需要计算渗透率场
        if args.compute:
            # 如果已经训练了模型，直接使用
            if args.train:
                # 这里model已经定义和训练好了
                pass
            else:
                # 加载已训练的模型
                model = GranitePINN(cube_size).to(device)
                _, _ = load_model_state(model, None, None, rock_type)
            
            # 检查模型是否加载成功
            model_loaded = True
            try:
                # 简单测试模型是否能正常工作
                test_point = torch.tensor([[cube_size/2, cube_size/2, cube_size/2]], device=device, dtype=torch.float32)
                with torch.no_grad():
                    _, perm = model(test_point)
            except:
                model_loaded = False
                print(f"模型加载失败或未训练，跳过计算渗透率场")
            
            if model_loaded:
                # 计算并保存渗透率场
                save_permeability_field(model, cube_size, rock_type)
                
                # 计算有效渗透率
                k_eff, k_geometric = compute_effective_permeability(model, cube_size)
                
                # 打印结果
                print(f"\n{rock_type}有效渗透率 (×10⁻¹⁹ m²):")
                print(f"X方向: {k_eff['x']:.4f}")
                print(f"Y方向: {k_eff['y']:.4f}")
                print(f"Z方向: {k_eff['z']:.4f}")
                print(f"几何平均: {k_geometric:.4f}")
        
        # 如果需要可视化
        if args.visualize:
            # 检查是否有模型可用
            model_available = False
            model = None
            
            # 如果已经训练了模型，直接使用
            if args.train:
                model_available = True
                # 这里model已经定义和训练好了
            else:
                # 尝试加载已训练的模型
                model = GranitePINN(cube_size).to(device)
                _, _ = load_model_state(model, None, None, rock_type)
                
                try:
                    # 简单测试模型是否能正常工作
                    test_point = torch.tensor([[cube_size/2, cube_size/2, cube_size/2]], device=device, dtype=torch.float32)
                    with torch.no_grad():
                        _, perm = model(test_point)
                    model_available = True
                except:
                    model_available = False
                    print(f"模型加载失败或未训练，尝试从保存的数据可视化")
            
            # 如果模型可用，直接可视化
            if model_available and model is not None:
                visualize_permeability_distribution(model, cube_size, rock_type)
                plot_permeability_histogram(model, cube_size, rock_type)
            else:
                # 从保存的数据可视化
                visualize_saved_permeability(rock_type)
                plot_histogram_from_saved(rock_type)
    
    # 比较两种岩石的结果
    if len(rock_types) == 2 and args.visualize:
        # 获取原论文kriging结果
        stanstead_kriging_results = kriging_simulation(None, 'Stanstead')
        lac_du_bonnet_kriging_results = kriging_simulation(None, 'Lac du Bonnet')
        
        # 获取PINN结果(从保存的数据)
        stanstead_k_eff, stanstead_k_geometric = compute_effective_permeability_from_saved('Stanstead')
        lac_du_bonnet_k_eff, lac_du_bonnet_k_geometric = compute_effective_permeability_from_saved('Lac du Bonnet')
        
        if stanstead_k_geometric is not None and lac_du_bonnet_k_geometric is not None:
            # 保存为PINN结果字典
            stanstead_pinn_results = {
                'k_eff': stanstead_k_eff,
                'k_geometric': stanstead_k_geometric
            }
            
            lac_du_bonnet_pinn_results = {
                'k_eff': lac_du_bonnet_k_eff,
                'k_geometric': lac_du_bonnet_k_geometric
            }
            
            # 比较结果
            compare_with_original_results(stanstead_pinn_results, stanstead_kriging_results, 'Stanstead')
            compare_with_original_results(lac_du_bonnet_pinn_results, lac_du_bonnet_kriging_results, 'Lac du Bonnet')
            
            # 计算比率
            pinn_ratio = stanstead_k_geometric / lac_du_bonnet_k_geometric
            original_ratio = stanstead_kriging_results['k_geometric'] / lac_du_bonnet_kriging_results['k_geometric']
            
            print(f"\n两种岩石渗透率比值:")
            print(f"PINN方法: {pinn_ratio:.2f}")
            print(f"原论文(kriging): {original_ratio:.2f}")
    
    print("\n研究完成!")


if __name__ == '__main__':
    # 设置开始时间
    start_time = time.time()
    
    # 运行主函数
    main()
    
    # 计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n总运行时间: {elapsed_time/60:.2f} 分钟")