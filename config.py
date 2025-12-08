"""
配置文件：存储所有超参数和路径配置
Video2Video 条件扩散模型 - 台风路径预测
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import torch

@dataclass
class DataConfig:
    """数据相关配置"""
    # 路径配置
    csv_path: str = "processed_typhoon_tracks.csv"
    era5_dir: str = "J:/Typhoon_data_final"
    
    # 网格尺寸
    grid_height: int = 44
    grid_width: int = 47
    
    # 时间分辨率（分钟）
    time_resolution_minutes: int = 30
    
    # 压力层（hPa）
    pressure_levels: List[int] = field(default_factory=lambda: [200, 500, 700, 850])
    
    # 3D ERA5 变量
    era5_3d_vars: List[str] = field(default_factory=lambda: ['z', 'r', 'q', 't', 'u', 'v', 'vo'])
    
    # 2D ERA5 变量
    era5_2d_vars: List[str] = field(default_factory=lambda: ['u10m', 'v10m', 't2m', 'tcwv', 'sp', 'msl'])


@dataclass
class ModelConfig:
    """模型相关配置"""
    # 序列长度
    t_cond: int = 12        # 条件帧长度（过去6小时 = 12 * 30min）
    t_future: int = 12      # 未来帧长度（未来6小时 = 12 * 30min）

    # === 输入维度 ===
    # 轨迹坐标维度
    coord_dim: int = 4      # lat, lon, vmax, pmin
    # 输出增量维度
    delta_dim: int = 3      # Δlat, Δlon, Δvmax

    # 条件特征维度（环境特征）
    cond_feature_dim: int = 64

    # === ERA5 编码器配置 ===
    era5_base_channels: int = 32
    era5_out_dim: int = 128    # ERA5 编码后的特征维度

    # === 轨迹编码器配置 ===
    coord_embed_dim: int = 64

    # === Transformer 配置 ===
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_ff_dim: int = 512
    dropout: float = 0.1

    # === 辅助 Heatmap Head 配置 ===
    use_heatmap_head: bool = True   # 是否使用辅助 heatmap 重建
    heatmap_loss_weight: float = 0.1  # heatmap 重建损失权重
    gaussian_sigma: float = 2.0      # 高斯斑点标准差

    # === Diffusion 配置 ===
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear" or "cosine"


@dataclass
class TrainConfig:
    """训练相关配置"""
    batch_size: int = 32  # 增大batch_size提高GPU利用率
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100

    # 数据加载优化
    num_workers: int = 0  # Windows下设为0避免卡顿
    pin_memory: bool = True  # 锁页内存加速传输
    use_amp: bool = False  # 混合精度训练（关闭避免nan）

    # 是否对真实帧和插值帧使用不同权重
    use_sample_weights: bool = True
    real_sample_weight: float = 1.0
    interp_sample_weight: float = 0.5

    # 早停机制
    early_stopping: bool = True
    patience: int = 10  # 连续N轮没改善就停止

    # 保存和日志
    save_interval: int = 10
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints/"

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_by: str = "storm_id"  # "storm_id" or "year"


@dataclass
class SampleConfig:
    """采样相关配置"""
    num_samples: int = 5          # 每个条件生成的样本数
    use_ddim: bool = True         # 是否使用 DDIM 加速采样
    ddim_steps: int = 50          # DDIM 采样步数
    eta: float = 0.0              # DDIM 随机性参数
    guidance_scale: float = 1.0   # 条件引导强度（若用 classifier-free guidance）


# 全局配置实例
data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
sample_cfg = SampleConfig()

