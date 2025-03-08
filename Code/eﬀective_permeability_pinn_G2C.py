import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
from itertools import product
import logging
import random
from scipy import stats

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class GranitePINN(nn.Module):
    """物理信息神经网络模型，完全匹配保存的模型结构"""
    
    def __init__(self, cube_size, hidden_layers=7, neurons_per_layer=128):  # 注意：增加了隐藏层数量
        super(GranitePINN, self).__init__()
        
        # 立方体大小（以毫米为单位）
        self.cube_size = cube_size
        
        # 特征提取层
        layers = []
        # 输入层: (x, y, z) 坐标
        input_size = 3
        
        # 隐藏层
        layers.append(nn.Linear(input_size, neurons_per_layer))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
            
        self.shared_net = nn.Sequential(*layers)
        
        # 压力头
        self.pressure_head = nn.Sequential(
            nn.Linear(neurons_per_layer, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 渗透率头
        self.perm_head = nn.Sequential(
            nn.Linear(neurons_per_layer, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        """前向传播"""
        x = x / self.cube_size  # 归一化坐标
        features = self.shared_net(x)
        
        # 压力和对数渗透率
        pressure = self.pressure_head(features)
        log_permeability = self.perm_head(features)
        permeability = torch.exp(log_permeability)  # 转换为实际渗透率
        
        return pressure, permeability, log_permeability
    
    def compute_pde_residual(self, x, p, K):
        """计算PDE残差: ∇·(K∇p) = 0"""
        # 确保需要梯度
        x.requires_grad_(True)
        
        # 重新计算压力以便获取梯度
        p_pred, _, _ = self.forward(x)
        
        # 计算压力梯度 ∇p
        p_grad = torch.autograd.grad(
            p_pred.sum(), x, create_graph=True, retain_graph=True
        )[0]
        
        # 计算 K∇p
        K_grad_p = K * p_grad
        
        # 计算 ∇·(K∇p)
        divergence = 0
        for i in range(3):  # x, y, z 方向
            div_i = torch.autograd.grad(
                K_grad_p[:, i].sum(), x, create_graph=True, retain_graph=True
            )[0][:, i]
            divergence += div_i
            
        return divergence



def generate_collocation_points(cube_size, n_points):
    """
    在立方体内部生成随机配点
    
    Args:
        cube_size: 立方体大小（毫米）
        n_points: 点数
        
    Returns:
        张量，形状 [n_points, 3]
    """
    points = torch.rand(n_points, 3) * cube_size
    return points

def generate_boundary_points(cube_size, n_points_per_side=9):
    """
    在立方体表面生成均匀分布的点
    
    Args:
        cube_size: 立方体大小（毫米）
        n_points_per_side: 每面的点数（将被平方）
        
    Returns:
        张量，形状 [6*n_points_per_side^2, 3]
    """
    # 每个维度上的离散点
    points_1d = torch.linspace(0, cube_size, n_points_per_side)
    
    boundary_points = []
    
    # 生成每个面上的点
    # 固定 x=0 和 x=cube_size 的面
    for x in [0, cube_size]:
        for y, z in product(points_1d, points_1d):
            boundary_points.append([x, y, z])
    
    # 固定 y=0 和 y=cube_size 的面
    for y in [0, cube_size]:
        for x, z in product(points_1d, points_1d):
            boundary_points.append([x, y, z])
            
    # 固定 z=0 和 z=cube_size 的面
    for z in [0, cube_size]:
        for x, y in product(points_1d, points_1d):
            boundary_points.append([x, y, z])
            
    return torch.tensor(boundary_points, dtype=torch.float32)

def compute_effective_permeability(model, cube_size, n_points=31, device='cpu'):
    """
    计算有效渗透率
    
    Args:
        model: 训练好的PINN模型
        cube_size: 立方体大小（毫米）
        n_points: 每个方向上的点数
        device: 使用的设备
        
    Returns:
        三个方向的有效渗透率和几何平均值
    """
    model.eval()
    
    # 创建网格点
    x = torch.linspace(0, cube_size, n_points).to(device)
    y = torch.linspace(0, cube_size, n_points).to(device)
    z = torch.linspace(0, cube_size, n_points).to(device)
    
    # 存储三个方向的渗透率
    K_eff_x = 0.0
    K_eff_y = 0.0
    K_eff_z = 0.0
    
    # X 方向的有效渗透率
    for j in range(n_points):
        for k in range(n_points):
            K_harmonic_sum = 0.0
            for i in range(n_points):
                point = torch.tensor([[x[i], y[j], z[k]]], dtype=torch.float32).to(device)
                _, K, _ = model(point)
                K_harmonic_sum += 1.0 / K.item()
            K_eff_x += n_points / K_harmonic_sum
    K_eff_x /= (n_points * n_points)
    
    # Y 方向的有效渗透率
    for i in range(n_points):
        for k in range(n_points):
            K_harmonic_sum = 0.0
            for j in range(n_points):
                point = torch.tensor([[x[i], y[j], z[k]]], dtype=torch.float32).to(device)
                _, K, _ = model(point)
                K_harmonic_sum += 1.0 / K.item()
            K_eff_y += n_points / K_harmonic_sum
    K_eff_y /= (n_points * n_points)
    
    # Z 方向的有效渗透率
    for i in range(n_points):
        for j in range(n_points):
            K_harmonic_sum = 0.0
            for k in range(n_points):
                point = torch.tensor([[x[i], y[j], z[k]]], dtype=torch.float32).to(device)
                _, K, _ = model(point)
                K_harmonic_sum += 1.0 / K.item()
            K_eff_z += n_points / K_harmonic_sum
    K_eff_z /= (n_points * n_points)
    
    # 几何平均有效渗透率
    K_eff_geometric = (K_eff_x * K_eff_y * K_eff_z) ** (1/3)
    
    return K_eff_x, K_eff_y, K_eff_z, K_eff_geometric

def plot_permeability_distributions(model, cube_size, rock_type, device='cpu'):
    """
    绘制渗透率分布
    
    Args:
        model: 训练好的PINN模型
        cube_size: 立方体大小（毫米）
        rock_type: 岩石类型
        device: 使用的设备
    """
    model.eval()
    
    # 创建保存目录
    save_dir = f'results/{rock_type}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建三个正交平面上的网格点
    n_points = 50
    x = torch.linspace(0, cube_size, n_points).to(device)
    y = torch.linspace(0, cube_size, n_points).to(device)
    z = torch.linspace(0, cube_size, n_points).to(device)
    
    # 存储三个平面上的渗透率
    K_xy = torch.zeros((n_points, n_points)).to(device)
    K_xz = torch.zeros((n_points, n_points)).to(device)
    K_yz = torch.zeros((n_points, n_points)).to(device)
    
    # 计算中心平面上的渗透率
    mid_z = cube_size / 2
    for i in range(n_points):
        for j in range(n_points):
            point = torch.tensor([[x[i], y[j], mid_z]], dtype=torch.float32).to(device)
            _, K, _ = model(point)
            K_xy[j, i] = K.item()
    
    mid_y = cube_size / 2
    for i in range(n_points):
        for k in range(n_points):
            point = torch.tensor([[x[i], mid_y, z[k]]], dtype=torch.float32).to(device)
            _, K, _ = model(point)
            K_xz[k, i] = K.item()
    
    mid_x = cube_size / 2
    for j in range(n_points):
        for k in range(n_points):
            point = torch.tensor([[mid_x, y[j], z[k]]], dtype=torch.float32).to(device)
            _, K, _ = model(point)
            K_yz[k, j] = K.item()
    
    # 确定共享的颜色范围
    vmin = min(K_xy.min().item(), K_xz.min().item(), K_yz.min().item())
    vmax = max(K_xy.max().item(), K_xz.max().item(), K_yz.max().item())
    
    # 绘制三个平面上的渗透率分布
    plt.figure(figsize=(15, 4))
    
    # XY平面
    plt.subplot(1, 3, 1)
    im = plt.imshow(K_xy.cpu().numpy(), extent=[0, cube_size, 0, cube_size], 
                   origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar(im, label=r'Permeability ($\times 10^{-19}$ m$^2$)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title(f'{rock_type} - XY Plane (z={mid_z:.0f}mm)')
    
    # XZ平面
    plt.subplot(1, 3, 2)
    im = plt.imshow(K_xz.cpu().numpy(), extent=[0, cube_size, 0, cube_size], 
                   origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar(im, label=r'Permeability ($\times 10^{-19}$ m$^2$)')
    plt.xlabel('X (mm)')
    plt.ylabel('Z (mm)')
    plt.title(f'{rock_type} - XZ Plane (y={mid_y:.0f}mm)')
    
    # YZ平面
    plt.subplot(1, 3, 3)
    im = plt.imshow(K_yz.cpu().numpy(), extent=[0, cube_size, 0, cube_size], 
                   origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar(im, label=r'Permeability ($\times 10^{-19}$ m$^2$)')
    plt.xlabel('Y (mm)')
    plt.ylabel('Z (mm)')
    plt.title(f'{rock_type} - YZ Plane (x={mid_x:.0f}mm)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/permeability_planes.png', dpi=300)
    plt.close()
    
    # 3D分布可视化
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    # 在整个体积中随机采样点
    n_samples = 1000
    X = torch.rand(n_samples) * cube_size
    Y = torch.rand(n_samples) * cube_size
    Z = torch.rand(n_samples) * cube_size
    points = torch.stack([X, Y, Z], dim=1).to(device)
    
    # 计算这些点的渗透率
    K_values = []
    for i in range(0, n_samples, 100):  # 批处理以避免内存问题
        batch = points[i:i+100]
        _, K, _ = model(batch)
        K_values.append(K.detach().cpu())
    
    K_values = torch.cat(K_values, dim=0).numpy()
    
    # 绘制3D散点图
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = plt.cm.viridis(norm(K_values))
    
    scatter = ax.scatter(X.numpy(), Y.numpy(), Z.numpy(), c=K_values, 
                         cmap='viridis', alpha=0.8, s=15)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'{rock_type} - 3D Permeability Distribution')
    
    cbar = plt.colorbar(scatter, ax=ax, label=r'Permeability ($\times 10^{-19}$ m$^2$)')
    
    plt.savefig(f'{save_dir}/permeability_3d.png', dpi=300)
    plt.close()
    
    # 绘制渗透率分布直方图
    plt.figure(figsize=(10, 6))
    
    # 收集更多样本进行直方图绘制
    n_samples = 10000
    X = torch.rand(n_samples) * cube_size
    Y = torch.rand(n_samples) * cube_size
    Z = torch.rand(n_samples) * cube_size
    points = torch.stack([X, Y, Z], dim=1).to(device)
    
    K_values = []
    for i in range(0, n_samples, 100):
        batch = points[i:i+100]
        _, K, _ = model(batch)
        K_values.append(K.detach().cpu())
    
    K_values = torch.cat(K_values, dim=0).numpy().flatten()
    
    # 计算统计量
    mean_k = np.mean(K_values)
    median_k = np.median(K_values)
    std_k = np.std(K_values)
    
    # 绘制直方图
    plt.hist(K_values, bins=50, density=True, alpha=0.6, color='skyblue')
    
    # 添加对数正态拟合
    x = np.linspace(min(K_values), max(K_values), 1000)
    shape, loc, scale = stats.lognorm.fit(K_values)
    pdf = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
    plt.plot(x, pdf, 'r-', linewidth=2, label='Log-normal Fit')
    
    # 添加均值和中位数线
    plt.axvline(mean_k, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_k:.2f}')
    plt.axvline(median_k, color='g', linestyle='--', linewidth=2, label=f'Median: {median_k:.2f}')
    
    # 添加注释框显示统计信息
    plt.text(0.05, 0.95, f'Mean: {mean_k:.2f}\nMedian: {median_k:.2f}\nStd Dev: {std_k:.2f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.xlabel(r'Permeability ($\times 10^{-19}$ m$^2$)')
    plt.ylabel('Probability Density')
    plt.title(f'{rock_type} Granite - Permeability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'{save_dir}/permeability_histogram.png', dpi=300)
    plt.close()


def load_model_state(model, optimizer, scheduler, rock_type, device='cpu'):
    """
    加载模型训练状态，增加错误处理和灵活性
    
    Args:
        model: PINN模型
        optimizer: 优化器
        scheduler: 学习率调度器
        rock_type: 岩石类型
        device: 当前设备
        
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
        logger.info(f"没有找到保存的模型状态: {model_path}")
        return 0, {'total': [], 'data': [], 'pde': [], 'boundary': [], 'reg': []}
    
    # 尝试加载模型权重
    try:
        logger.info(f"正在从 {model_path} 加载模型...")
        
        # 加载模型权重并处理可能的结构不匹配
        state_dict = torch.load(model_path, map_location=device)
        
        # 尝试直接加载（可能会失败）
        try:
            model.load_state_dict(state_dict)
            logger.info("成功加载模型权重")
        except Exception as e:
            logger.warning(f"直接加载模型权重失败: {str(e)}")
            logger.info("尝试使用灵活加载策略...")
            
            # 灵活加载: 过滤掉不匹配的键
            model_dict = model.state_dict()
            
            # 首先尝试映射层名称
            name_mapping = {
                'shared_net': 'feature_layers',
                'perm_head': 'permeability_head'
            }
            
            mapped_state_dict = {}
            for old_key, value in state_dict.items():
                new_key = old_key
                for old_name, new_name in name_mapping.items():
                    if old_name in new_key:
                        new_key = new_key.replace(old_name, new_name)
                mapped_state_dict[new_key] = value
            
            # 过滤形状不匹配的键
            filtered_state_dict = {k: v for k, v in mapped_state_dict.items() 
                                if k in model_dict and v.shape == model_dict[k].shape}
            
            # 记录加载统计
            missing_keys = set(model_dict.keys()) - set(filtered_state_dict.keys())
            unexpected_keys = set(mapped_state_dict.keys()) - set(model_dict.keys())
            
            if missing_keys:
                logger.warning(f"以下键在保存模型中缺失: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"以下键在当前模型中不存在: {unexpected_keys}")
            
            # 尝试加载过滤后的权重
            try:
                model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"成功加载部分模型权重 ({len(filtered_state_dict)}/{len(model_dict)} 键)")
            except Exception as inner_e:
                logger.error(f"加载过滤后的权重仍然失败: {str(inner_e)}")
                logger.info("将使用新初始化的模型")
                return 0, {'total': [], 'data': [], 'pde': [], 'boundary': [], 'reg': []}
    except Exception as e:
        logger.error(f"加载模型权重过程中出现错误: {str(e)}")
        return 0, {'total': [], 'data': [], 'pde': [], 'boundary': [], 'reg': []}
    
    # 加载训练状态
    try:
        logger.info(f"正在从 {training_path} 加载训练状态...")
        training_state = torch.load(training_path, map_location=device)
        epoch = training_state.get('epoch', 0)
        
        # 加载优化器状态（如果存在）
        if optimizer and 'optimizer_state' in training_state:
            try:
                # 处理优化器状态中可能存在的设备不匹配问题
                optimizer_state = training_state['optimizer_state']
                
                # 如果优化器状态包含张量，确保它们在正确的设备上
                for param_group in optimizer_state['param_groups']:
                    for k, v in param_group.items():
                        if isinstance(v, torch.Tensor):
                            param_group[k] = v.to(device)
                
                for state in optimizer_state['state'].values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                
                optimizer.load_state_dict(optimizer_state)
                logger.info("成功加载优化器状态")
            except Exception as e:
                logger.warning(f"加载优化器状态失败: {str(e)}")
        
        # 加载调度器状态（如果存在）
        if scheduler and 'scheduler_state' in training_state:
            try:
                scheduler.load_state_dict(training_state['scheduler_state'])
                logger.info("成功加载学习率调度器状态")
            except Exception as e:
                logger.warning(f"加载学习率调度器状态失败: {str(e)}")
        
        # 加载损失历史
        loss_history = training_state.get('loss_history', {'total': [], 'data': [], 'pde': [], 'boundary': [], 'reg': []})
        
    except Exception as e:
        logger.error(f"加载训练状态失败: {str(e)}")
        epoch = 0
        loss_history = {'total': [], 'data': [], 'pde': [], 'boundary': [], 'reg': []}
    
    logger.info(f"已从第 {epoch} 轮恢复模型训练状态")
    return epoch, loss_history


def save_model_state(model, optimizer, scheduler, epoch, loss_history, rock_type):
    """
    保存模型训练状态
    
    Args:
        model: PINN模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮数
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
    logger.info(f"已保存模型训练状态到 {save_dir}")

def plot_loss_history(loss_history, rock_type):
    """
    绘制损失历史
    
    Args:
        loss_history: 损失历史字典
        rock_type: 岩石类型
    """
    # 创建保存目录
    save_dir = f'results/{rock_type}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制总损失
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history['total'], 'b-', linewidth=2, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'{rock_type} Granite - Total Loss History')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/total_loss.png', dpi=300)
    plt.close()
    
    # 绘制各分量损失
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history['data'], 'b-', linewidth=2, label='Data Loss')
    plt.semilogy(loss_history['pde'], 'r-', linewidth=2, label='PDE Loss')
    plt.semilogy(loss_history['boundary'], 'g-', linewidth=2, label='Boundary Loss')
    plt.semilogy(loss_history['reg'], 'c-', linewidth=2, label='Reg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'{rock_type} Granite - Component Loss History')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/component_loss.png', dpi=300)
    plt.close()

def train_model(model, rock_type, cube_size, n_epochs=50000, batch_size=1024, device='cpu', 
                resume_training=True, save_interval=1000):
    """
    训练PINN模型
    
    Args:
        model: PINN模型
        rock_type: 岩石类型
        cube_size: 立方体大小（毫米）
        n_epochs: 训练轮数
        batch_size: 批次大小
        device: 设备
        resume_training: 是否恢复训练
        save_interval: 保存间隔
        
    Returns:
        训练好的模型和损失历史
    """
    logger.info(f"开始训练 {rock_type} 花岗岩模型，使用设备: {device}")
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, verbose=True)
    
    # 加载模型状态（如果存在）
    if resume_training:
        start_epoch, loss_history = load_model_state(model, optimizer, scheduler, rock_type, device)
    else:
        start_epoch = 0
        loss_history = {'total': [], 'data': [], 'pde': [], 'boundary': [], 'reg': []}
    
    # 生成训练点
    # 表面测量点（通常从实验获得）
    boundary_points = generate_boundary_points(cube_size, n_points_per_side=9).to(device)
    
    # 根据岩石类型设置表面渗透率
    if rock_type == 'Stanstead':
        # Stanstead花岗岩的表面渗透率（约59×10⁻¹⁹m²）
        boundary_K = 59.0 * torch.ones(boundary_points.shape[0], 1).to(device) * 1e-19
    elif rock_type == 'LacDuBonnet':
        # Lac du Bonnet花岗岩的表面渗透率（约1.09×10⁻¹⁹m²）
        boundary_K = 1.09 * torch.ones(boundary_points.shape[0], 1).to(device) * 1e-19
    else:
        # 默认值
        boundary_K = 10.0 * torch.ones(boundary_points.shape[0], 1).to(device) * 1e-19
        logger.warning(f"未知岩石类型: {rock_type}，使用默认渗透率10.0×10⁻¹⁹m²")
    
    # 训练代码保持不变...
    # [训练循环和其他代码保持不变]
    
    # 内部配点数
    n_collocation = batch_size
    
    # 损失权重
    data_weight = 10.0
    pde_weight = 1.0
    boundary_weight = 1.0
    reg_weight = 0.01
    
    # 训练循环
    model.train()
    start_time = time.time()
    
    for epoch in range(start_epoch, n_epochs):
        # 生成内部配点
        collocation_points = generate_collocation_points(cube_size, n_collocation).to(device)
        
        # 前向传播
        # 边界点
        _, boundary_K_pred, boundary_logK_pred = model(boundary_points)
        
        # 内部点
        pressure_pred, K_pred, _ = model(collocation_points)
        
        # 计算PDE残差
        pde_residual = model.compute_pde_residual(collocation_points, pressure_pred, K_pred)
        
        # 损失计算
        # 数据损失 - 表面渗透率拟合
        data_loss = torch.mean((boundary_K_pred - boundary_K) ** 2)
        
        # PDE损失 - 达西定律
        pde_loss = torch.mean(pde_residual ** 2)
        
        # 边界损失 - 渗透率平滑过渡
        # 简化为相邻点渗透率差异的平方
        boundary_loss = 0.0
        if boundary_points.shape[0] > 1:
            # 计算所有点对之间的差异，简化为只考虑相邻点
            for i in range(boundary_points.shape[0] - 1):
                boundary_loss += torch.mean((boundary_logK_pred[i] - boundary_logK_pred[i+1]) ** 2)
            boundary_loss /= (boundary_points.shape[0] - 1)
        
        # 正则化损失 - L2范数
        reg_loss = sum(torch.sum(param ** 2) for param in model.parameters())
        
        # 总损失
        total_loss = (data_weight * data_loss + 
                      pde_weight * pde_loss + 
                      boundary_weight * boundary_loss + 
                      reg_weight * reg_loss)
        
        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 学习率调度
        scheduler.step(total_loss)
        
        # 记录损失
        loss_history['total'].append(total_loss.item())
        loss_history['data'].append(data_loss.item())
        loss_history['pde'].append(pde_loss.item())
        loss_history['boundary'].append(boundary_loss.item())
        loss_history['reg'].append(reg_loss.item())
        
        # 打印进度
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch}/{n_epochs}, Total Loss: {total_loss:.6f}, Time: {elapsed:.2f}s")
            logger.info(f"  Data Loss: {data_loss:.6f}, PDE Loss: {pde_loss:.6f}, "
                        f"Boundary Loss: {boundary_loss:.6f}, Reg Loss: {reg_loss:.6f}")
            start_time = time.time()
        
        # 定期保存模型
        if epoch % save_interval == 0 and epoch > 0:
            save_model_state(model, optimizer, scheduler, epoch, loss_history, rock_type)
            
    # 最终保存
    save_model_state(model, optimizer, scheduler, n_epochs, loss_history, rock_type)
    
    return model, loss_history



def create_matching_model(model_path, cube_size, device='cpu'):
    """
    创建与保存模型结构匹配的新模型
    
    Args:
        model_path: 模型权重文件路径
        cube_size: 立方体大小
        device: 设备
        
    Returns:
        与保存模型结构匹配的新模型实例
    """
    # 检查保存模型结构
    model_info = inspect_saved_model(model_path, map_location=device)
    logger.info(f"保存的模型结构: {model_info}")
    
    # 根据检查结果创建匹配的模型
    hidden_layers = model_info['hidden_layers']
    neurons_per_layer = model_info['neurons_per_layer']
    
    logger.info(f"创建匹配模型，隐藏层数: {hidden_layers}，每层神经元数: {neurons_per_layer}")
    model = GranitePINN(cube_size, hidden_layers=hidden_layers, neurons_per_layer=neurons_per_layer).to(device)
    
    return model

def main():
    """主函数 - 处理两种花岗岩类型"""
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 岩石参数
    rock_types = {
        'Stanstead': {
            'cube_size': 280.0,  # mm
            'expected_k': 59.3e-19  # m²
        },
        'LacDuBonnet': {
            'cube_size': 300.0,  # mm
            'expected_k': 1.09e-19  # m²
        }
    }
    
    # 结果汇总表
    summary_results = {}
    
    # 处理每种岩石类型
    for rock_type, rock_params in rock_types.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"开始处理 {rock_type} 花岗岩")
        logger.info(f"{'='*50}")
        
        cube_size = rock_params['cube_size']
        
        # 检查保存的模型路径
        model_path = f'models/{rock_type}/model_weights.pth'
        
        # 创建模型
        if os.path.exists(model_path):
            # 尝试根据保存的模型创建匹配的结构
            try:
                model_info = inspect_saved_model(model_path, map_location=device)
                hidden_layers = model_info.get('hidden_layers', 7)  # 默认7层
                neurons_per_layer = model_info.get('neurons_per_layer', 128)  # 默认128个神经元
                
                logger.info(f"创建匹配模型，隐藏层数: {hidden_layers}，每层神经元数: {neurons_per_layer}")
                model = GranitePINN(cube_size, hidden_layers=hidden_layers, neurons_per_layer=neurons_per_layer).to(device)
            except Exception as e:
                logger.warning(f"创建匹配模型失败: {str(e)}，将使用默认模型结构")
                model = GranitePINN(cube_size, hidden_layers=7, neurons_per_layer=128).to(device)
        else:
            # 创建新模型
            model = GranitePINN(cube_size, hidden_layers=7, neurons_per_layer=128).to(device)
            logger.info(f"创建了新的 {rock_type} 花岗岩PINN模型")
        
        # 训练模型
        model, loss_history = train_model(
            model=model,
            rock_type=rock_type,
            cube_size=cube_size,
            n_epochs=50000,  # 可以根据需要调整
            device=device,
            resume_training=True,
            save_interval=1000
        )
        
        # 绘制损失历史
        plot_loss_history(loss_history, rock_type)
        
        # 计算有效渗透率
        K_eff_x, K_eff_y, K_eff_z, K_eff_geometric = compute_effective_permeability(
            model, cube_size, device=device
        )
        
        # 转换为正确的单位（×10⁻¹⁹m²）
        K_eff_x_scaled = K_eff_x * 1e19
        K_eff_y_scaled = K_eff_y * 1e19
        K_eff_z_scaled = K_eff_z * 1e19
        K_eff_geometric_scaled = K_eff_geometric * 1e19
        
        # 记录结果
        summary_results[rock_type] = {
            'K_eff_x': K_eff_x_scaled,
            'K_eff_y': K_eff_y_scaled,
            'K_eff_z': K_eff_z_scaled,
            'K_eff_geometric': K_eff_geometric_scaled,
            'expected_k': rock_params['expected_k'] * 1e19,
            'relative_error': abs(K_eff_geometric_scaled - rock_params['expected_k'] * 1e19) / (rock_params['expected_k'] * 1e19) * 100
        }
        
        # 输出结果
        logger.info(f"\n计算 {rock_type} 花岗岩有效渗透率 (×10⁻¹⁹m²):")
        logger.info(f"X 方向: {K_eff_x_scaled:.2f}")
        logger.info(f"Y 方向: {K_eff_y_scaled:.2f}")
        logger.info(f"Z 方向: {K_eff_z_scaled:.2f}")
        logger.info(f"几何平均: {K_eff_geometric_scaled:.2f}")
        
        # 对比原论文结果
        expected_k = rock_params['expected_k'] * 1e19
        relative_error = abs(K_eff_geometric_scaled - expected_k) / expected_k * 100
        logger.info(f"原论文结果: {expected_k:.2f} ×10⁻¹⁹m²")
        logger.info(f"相对误差: {relative_error:.2f}%")
        
        # 绘制渗透率分布
        plot_permeability_distributions(model, cube_size, rock_type, device=device)
    
    # 输出汇总结果
    logger.info("\n" + "="*60)
    logger.info("两种花岗岩有效渗透率汇总结果 (×10⁻¹⁹m²):")
    logger.info("="*60)
    
    logger.info(f"{'岩石类型':<15} {'X方向':<10} {'Y方向':<10} {'Z方向':<10} {'几何平均':<10} {'原论文':<10} {'相对误差':<10}")
    logger.info("-"*75)
    
    for rock_type, results in summary_results.items():
        logger.info(f"{rock_type:<15} {results['K_eff_x']:<10.2f} {results['K_eff_y']:<10.2f} "
                   f"{results['K_eff_z']:<10.2f} {results['K_eff_geometric']:<10.2f} "
                   f"{results['expected_k']:<10.2f} {results['relative_error']:<10.2f}%")
    
    # 计算两种花岗岩渗透率比值
    if 'Stanstead' in summary_results and 'LacDuBonnet' in summary_results:
        stanstead_k = summary_results['Stanstead']['K_eff_geometric']
        lacbonnet_k = summary_results['LacDuBonnet']['K_eff_geometric']
        ratio = stanstead_k / lacbonnet_k
        
        expected_stanstead_k = rock_types['Stanstead']['expected_k'] * 1e19
        expected_lacbonnet_k = rock_types['LacDuBonnet']['expected_k'] * 1e19
        expected_ratio = expected_stanstead_k / expected_lacbonnet_k
        
        ratio_error = abs(ratio - expected_ratio) / expected_ratio * 100
        
        logger.info("\n" + "-"*75)
        logger.info(f"Stanstead/LacDuBonnet 渗透率比值: {ratio:.2f}")
        logger.info(f"原论文比值: {expected_ratio:.2f}")
        logger.info(f"比值相对误差: {ratio_error:.2f}%")
    
    logger.info("\n完成!")

# 添加inspect_saved_model函数
def inspect_saved_model(model_path, map_location='cpu'):
    """
    检查保存模型的结构
    
    Args:
        model_path: 模型权重文件路径
        map_location: 设备映射
        
    Returns:
        字典，包含模型结构信息
    """
    state_dict = torch.load(model_path, map_location=map_location)
    
    # 提取网络层信息
    layer_info = {}
    for key in state_dict.keys():
        layer_info[key] = state_dict[key].shape
        
    # 分析网络结构
    hidden_layers = 0
    neurons_per_layer = 0
    
    for key in layer_info:
        if 'shared_net' in key and '.weight' in key:
            layer_num = int(key.split('.')[1])
            if layer_num % 2 == 0:  # Linear层
                hidden_layers = max(hidden_layers, layer_num // 2 + 1)
                if neurons_per_layer == 0 and layer_info[key].shape[0] > 3:  # 排除输入层
                    neurons_per_layer = layer_info[key][0]
    
    return {
        'hidden_layers': hidden_layers,
        'neurons_per_layer': neurons_per_layer,
        'layer_info': layer_info
    }
    

def main():
    """主函数 - 处理两种花岗岩类型"""
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 岩石参数
    rock_types = {
        'Stanstead': {
            'cube_size': 280.0,  # mm
            'expected_k': 59.3e-19  # m²
        },
        'LacDuBonnet': {
            'cube_size': 300.0,  # mm
            'expected_k': 1.09e-19  # m²
        }
    }
    
    # 结果汇总表
    summary_results = {}
    
    # 处理每种岩石类型
    for rock_type, rock_params in rock_types.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"开始处理 {rock_type} 花岗岩")
        logger.info(f"{'='*50}")
        
        cube_size = rock_params['cube_size']
        
        # 检查保存的模型路径
        model_path = f'models/{rock_type}/model_weights.pth'
        
        # 创建模型
        if os.path.exists(model_path):
            # 尝试根据保存的模型创建匹配的结构
            try:
                model_info = inspect_saved_model(model_path, map_location=device)
                hidden_layers = model_info.get('hidden_layers', 7)  # 默认7层
                neurons_per_layer = model_info.get('neurons_per_layer', 128)  # 默认128个神经元
                
                logger.info(f"创建匹配模型，隐藏层数: {hidden_layers}，每层神经元数: {neurons_per_layer}")
                model = GranitePINN(cube_size, hidden_layers=hidden_layers, neurons_per_layer=neurons_per_layer).to(device)
            except Exception as e:
                logger.warning(f"创建匹配模型失败: {str(e)}，将使用默认模型结构")
                model = GranitePINN(cube_size, hidden_layers=7, neurons_per_layer=128).to(device)
        else:
            # 创建新模型
            model = GranitePINN(cube_size, hidden_layers=7, neurons_per_layer=128).to(device)
            logger.info(f"创建了新的 {rock_type} 花岗岩PINN模型")
        
        # 训练模型
        model, loss_history = train_model(
            model=model,
            rock_type=rock_type,
            cube_size=cube_size,
            n_epochs=50000,  # 可以根据需要调整
            device=device,
            resume_training=True,
            save_interval=1000
        )
        
        # 绘制损失历史
        plot_loss_history(loss_history, rock_type)
        
        # 计算有效渗透率
        K_eff_x, K_eff_y, K_eff_z, K_eff_geometric = compute_effective_permeability(
            model, cube_size, device=device
        )
        
        # 转换为正确的单位（×10⁻¹⁹m²）
        K_eff_x_scaled = K_eff_x * 1e19
        K_eff_y_scaled = K_eff_y * 1e19
        K_eff_z_scaled = K_eff_z * 1e19
        K_eff_geometric_scaled = K_eff_geometric * 1e19
        
        # 记录结果
        summary_results[rock_type] = {
            'K_eff_x': K_eff_x_scaled,
            'K_eff_y': K_eff_y_scaled,
            'K_eff_z': K_eff_z_scaled,
            'K_eff_geometric': K_eff_geometric_scaled,
            'expected_k': rock_params['expected_k'] * 1e19,
            'relative_error': abs(K_eff_geometric_scaled - rock_params['expected_k'] * 1e19) / (rock_params['expected_k'] * 1e19) * 100
        }
        
        # 输出结果
        logger.info(f"\n计算 {rock_type} 花岗岩有效渗透率 (×10⁻¹⁹m²):")
        logger.info(f"X 方向: {K_eff_x_scaled:.2f}")
        logger.info(f"Y 方向: {K_eff_y_scaled:.2f}")
        logger.info(f"Z 方向: {K_eff_z_scaled:.2f}")
        logger.info(f"几何平均: {K_eff_geometric_scaled:.2f}")
        
        # 对比原论文结果
        expected_k = rock_params['expected_k'] * 1e19
        relative_error = abs(K_eff_geometric_scaled - expected_k) / expected_k * 100
        logger.info(f"原论文结果: {expected_k:.2f} ×10⁻¹⁹m²")
        logger.info(f"相对误差: {relative_error:.2f}%")
        
        # 绘制渗透率分布
        plot_permeability_distributions(model, cube_size, rock_type, device=device)
    
    # 输出汇总结果
    logger.info("\n" + "="*60)
    logger.info("两种花岗岩有效渗透率汇总结果 (×10⁻¹⁹m²):")
    logger.info("="*60)
    
    logger.info(f"{'岩石类型':<15} {'X方向':<10} {'Y方向':<10} {'Z方向':<10} {'几何平均':<10} {'原论文':<10} {'相对误差':<10}")
    logger.info("-"*75)
    
    for rock_type, results in summary_results.items():
        logger.info(f"{rock_type:<15} {results['K_eff_x']:<10.2f} {results['K_eff_y']:<10.2f} "
                   f"{results['K_eff_z']:<10.2f} {results['K_eff_geometric']:<10.2f} "
                   f"{results['expected_k']:<10.2f} {results['relative_error']:<10.2f}%")
    
    # 计算两种花岗岩渗透率比值
    if 'Stanstead' in summary_results and 'LacDuBonnet' in summary_results:
        stanstead_k = summary_results['Stanstead']['K_eff_geometric']
        lacbonnet_k = summary_results['LacDuBonnet']['K_eff_geometric']
        ratio = stanstead_k / lacbonnet_k
        
        expected_stanstead_k = rock_types['Stanstead']['expected_k'] * 1e19
        expected_lacbonnet_k = rock_types['LacDuBonnet']['expected_k'] * 1e19
        expected_ratio = expected_stanstead_k / expected_lacbonnet_k
        
        ratio_error = abs(ratio - expected_ratio) / expected_ratio * 100
        
        logger.info("\n" + "-"*75)
        logger.info(f"Stanstead/LacDuBonnet 渗透率比值: {ratio:.2f}")
        logger.info(f"原论文比值: {expected_ratio:.2f}")
        logger.info(f"比值相对误差: {ratio_error:.2f}%")
    
    logger.info("\n完成!")

# 添加inspect_saved_model函数
def inspect_saved_model(model_path, map_location='cpu'):
    """
    检查保存模型的结构
    
    Args:
        model_path: 模型权重文件路径
        map_location: 设备映射
        
    Returns:
        字典，包含模型结构信息
    """
    state_dict = torch.load(model_path, map_location=map_location)
    
    # 提取网络层信息
    layer_info = {}
    for key in state_dict.keys():
        layer_info[key] = state_dict[key].shape
        
    # 分析网络结构
    hidden_layers = 0
    neurons_per_layer = 0
    
    for key in layer_info:
        if 'shared_net' in key and '.weight' in key:
            layer_num = int(key.split('.')[1])
            if layer_num % 2 == 0:  # Linear层
                hidden_layers = max(hidden_layers, layer_num // 2 + 1)
                if neurons_per_layer == 0 and layer_info[key].shape[0] > 3:  # 排除输入层
                    neurons_per_layer = layer_info[key][0]
    
    return {
        'hidden_layers': hidden_layers,
        'neurons_per_layer': neurons_per_layer,
        'layer_info': layer_info
    }    
    
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("程序执行中发生错误:")