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
    """模型相关配置 - 针对 RTX 4090 24GB 优化"""
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

    # === ERA5 编码器配置（增强） ===
    era5_base_channels: int = 64   # 32 → 64，更强的特征提取
    era5_out_dim: int = 256        # 128 → 256

    # === 轨迹编码器配置 ===
    coord_embed_dim: int = 128     # 64 → 128

    # === Transformer 配置（大幅增强） ===
    transformer_dim: int = 512     # 384 → 512
    transformer_heads: int = 8
    transformer_layers: int = 8    # 6 → 8 层
    transformer_ff_dim: int = 2048 # 1024 → 2048
    dropout: float = 0.1

    # === 辅助 Heatmap Head 配置 ===
    use_heatmap_head: bool = True
    heatmap_loss_weight: float = 0.2
    gaussian_sigma: float = 2.0

    # === Diffusion 配置 ===
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"


@dataclass
class TrainConfig:
    """训练相关配置 - 针对 RTX 4090 24GB 优化"""
    batch_size: int = 64          # 32 → 64，4090 显存充足
    learning_rate: float = 1e-4   # 降低学习率避免 NaN
    weight_decay: float = 1e-5
    num_epochs: int = 200         # 更多训练轮数

    # 数据加载优化
    num_workers: int = 4          # 4090 搭配的 CPU 通常更强
    pin_memory: bool = True
    use_amp: bool = False         # 先关闭 AMP 排查问题

    # 是否对真实帧和插值帧使用不同权重
    use_sample_weights: bool = True
    real_sample_weight: float = 1.0
    interp_sample_weight: float = 0.5

    # 早停机制
    early_stopping: bool = True
    patience: int = 25            # 更大耐心

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
    split_by: str = "storm_id"
    
    # 学习率调度
    lr_scheduler: str = "cosine_warmup"
    warmup_epochs: int = 10       # 增加 warmup
    
    # 梯度累积（可选，如果想用更大 batch）
    gradient_accumulation_steps: int = 1


@dataclass
class SampleConfig:
    """采样相关配置 - 针对 RTX 4090 优化"""
    num_samples: int = 10         # 5 → 10，更多集合成员提高稳定性
    use_ddim: bool = True
    ddim_steps: int = 50
    eta: float = 0.0
    guidance_scale: float = 1.0


# 全局配置实例
data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
sample_cfg = SampleConfig()

