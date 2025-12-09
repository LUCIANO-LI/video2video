"""
混合式条件扩散模型：直接生成坐标增量 + 辅助 Heatmap 重建
Video2Video 条件扩散模型 - 台风路径预测

架构：
- 输入: 过去轨迹坐标 (T_cond, 4) + ERA5视频 (T_cond, C, H, W) + 环境特征 (T_cond, D)
- 输出: 未来坐标增量 (T_future, 3) [Δlat, Δlon, Δvmax] + 辅助heatmap
- 扩散过程在坐标增量空间进行
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

from config import model_cfg, data_cfg


# ============== 基础模块 ==============

class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class PositionalEncoding(nn.Module):
    """序列位置编码"""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ============== 编码器模块 ==============

class CoordEncoder(nn.Module):
    """轨迹坐标编码器 - 增强版"""
    def __init__(self, coord_dim: int = 4, embed_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


class ERA5Encoder(nn.Module):
    """ERA5 视频编码器 - 增强版（针对 RTX 4090 优化）"""
    def __init__(self, in_channels: int = 34, out_dim: int = 256, base_ch: int = 64):
        super().__init__()
        
        # 第一阶段：通道压缩 + 特征提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
        )
        
        # 第二阶段：下采样 + 特征增强
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
        )
        
        # 第三阶段：进一步下采样
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.GELU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.GELU(),
        )
        
        # 第四阶段：更深的特征
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.GELU(),
        )
        
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(base_ch * 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 通道注意力 (SE-like)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, base_ch),
            nn.ReLU(),
            nn.Linear(base_ch, base_ch * 4),
            nn.Sigmoid()
        )
        
        # 全局池化 + 投影
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(base_ch * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, era5: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = era5.shape
        x = era5.view(B * T, C, H, W)
        
        # 特征提取
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 注意力机制
        spatial_w = self.spatial_attn(x)
        x = x * spatial_w
        
        channel_w = self.channel_attn(x).view(B * T, -1, 1, 1)
        x = x * channel_w
        
        # 池化 + 投影
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        
        return x.view(B, T, -1)


class FeatureEncoder(nn.Module):
    """环境特征编码器 - 增强版"""
    def __init__(self, in_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class ConditionFusion(nn.Module):
    """融合所有条件输入 - 简化稳定版"""
    def __init__(self, coord_dim: int, era5_dim: int, feat_dim: int, out_dim: int):
        super().__init__()
        total_dim = coord_dim + era5_dim + feat_dim
        
        # 简单拼接 + 投影（更稳定）
        self.proj = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, coord_emb, era5_emb, feat_emb) -> torch.Tensor:
        x = torch.cat([coord_emb, era5_emb, feat_emb], dim=-1)
        return self.proj(x)


# ============== 去噪网络（Denoiser） ==============

class CrossAttentionBlock(nn.Module):
    """Cross-Attention 模块：让目标序列关注条件序列"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, T_future, d_model), context: (B, T_cond, d_model)
        attn_out, _ = self.cross_attn(x, context, context)
        return self.norm(x + self.dropout(attn_out))


class TransformerDenoiserBlock(nn.Module):
    """单个 Transformer Denoiser 块：Self-Attention + Cross-Attention + FFN"""
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = CrossAttentionBlock(d_model, n_heads, dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention with condition
        x = self.cross_attn(x, context)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerDenoiser(nn.Module):
    """
    改进的 Transformer-based 去噪网络
    - 使用 Cross-Attention 替代简单拼接
    - 添加时间步自适应层归一化 (AdaLN)
    - 增加残差连接
    """
    def __init__(
        self,
        delta_dim: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,  # 增加层数
        ff_dim: int = 1024,  # 增加 FFN 维度
        dropout: float = 0.1,
        t_future: int = 12
    ):
        super().__init__()
        self.d_model = d_model
        self.t_future = t_future
        self.delta_dim = delta_dim

        # 增量序列嵌入（增强）
        self.delta_embed = nn.Sequential(
            nn.Linear(delta_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 时间步嵌入（增强）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # 时间步自适应调制（AdaLN-like）
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),  # scale and shift
        )

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len=100)
        
        # 可学习的 lead time 编码（未来时间步位置）
        self.lead_time_embed = nn.Parameter(torch.randn(1, t_future, d_model) * 0.02)

        # Transformer Denoiser 块
        self.blocks = nn.ModuleList([
            TransformerDenoiserBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出投影（增强）
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, delta_dim)
        )
        
        # 残差连接的 skip projection
        self.skip_proj = nn.Linear(delta_dim, delta_dim)

    def forward(
        self,
        noisy_deltas: torch.Tensor,  # (B, T_future, delta_dim)
        t: torch.Tensor,              # (B,) diffusion timestep
        cond_embed: torch.Tensor      # (B, T_cond, d_model)
    ) -> torch.Tensor:
        B = noisy_deltas.shape[0]

        # 嵌入噪声增量
        x = self.delta_embed(noisy_deltas)  # (B, T_future, d_model)
        
        # 添加位置编码 + lead time 编码
        x = self.pos_enc(x)
        x = x + self.lead_time_embed
        
        # 时间步嵌入
        t_emb = self.time_embed(t)  # (B, d_model)
        
        # 时间步自适应调制
        t_mlp = self.time_mlp(t_emb)  # (B, d_model * 2)
        scale, shift = t_mlp.chunk(2, dim=-1)  # 各 (B, d_model)
        # 限制 scale 范围避免数值爆炸
        scale = torch.tanh(scale) * 0.5  # scale 在 [-0.5, 0.5]
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 通过 Transformer 块（带 Cross-Attention）
        for block in self.blocks:
            x = block(x, cond_embed)

        # 输出投影
        out = self.out_proj(x)  # (B, T_future, delta_dim)
        
        # 残差连接（使用较小的权重）
        out = out + 0.1 * self.skip_proj(noisy_deltas)
        
        return out


# ============== 辅助 Heatmap Head ==============

class HeatmapHead(nn.Module):
    """
    辅助 Heatmap 重建头
    从预测的坐标生成 heatmap（弱监督）
    """
    def __init__(self, grid_h: int = 44, grid_w: int = 47, sigma: float = 2.0):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.sigma = sigma

        # 预计算网格
        y = torch.arange(grid_h).float()
        x = torch.arange(grid_w).float()
        self.register_buffer('grid_y', y.view(1, 1, grid_h, 1))
        self.register_buffer('grid_x', x.view(1, 1, 1, grid_w))

    def forward(
        self,
        lat: torch.Tensor,    # (B, T) 归一化到 [0, grid_h-1]
        lon: torch.Tensor,    # (B, T) 归一化到 [0, grid_w-1]
        intensity: torch.Tensor = None  # (B, T) 可选的强度
    ) -> torch.Tensor:
        """生成高斯 heatmap"""
        B, T = lat.shape

        # (B, T, 1, 1)
        lat = lat.view(B, T, 1, 1)
        lon = lon.view(B, T, 1, 1)

        # 计算高斯
        dist_sq = (self.grid_y - lat) ** 2 + (self.grid_x - lon) ** 2
        heatmap = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        if intensity is not None:
            intensity = intensity.view(B, T, 1, 1)
            heatmap = heatmap * intensity

        return heatmap  # (B, T, H, W)



# ============== DDPM 调度器 ==============

class DDPMScheduler:
    """DDPM 噪声调度器 - 支持 DDIM 加速采样"""

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine"  # 改为 cosine 默认
    ):
        self.num_steps = num_steps

        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif beta_schedule == "cosine":
            # 改进的 cosine schedule
            steps = num_steps + 1
            s = 0.008
            x = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.999)
        elif beta_schedule == "squaredcos_cap_v2":
            # 更平滑的 cosine schedule
            steps = num_steps + 1
            t = torch.linspace(0, num_steps, steps) / num_steps
            alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 用于 x0 预测
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy_x = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return noisy_x, noise
    
    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """从噪声预测原始数据 x0"""
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t]
        
        while sqrt_recip.dim() < x_t.dim():
            sqrt_recip = sqrt_recip.unsqueeze(-1)
            sqrt_recipm1 = sqrt_recipm1.unsqueeze(-1)
        
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def step(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor):
        """DDPM 反向一步"""
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]

        while coef1.dim() < x_t.dim():
            coef1 = coef1.unsqueeze(-1)
            coef2 = coef2.unsqueeze(-1)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        while sqrt_alpha.dim() < x_t.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        x_0_pred = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        
        # Clip x0 预测值以提高稳定性
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        mean = coef1 * x_0_pred + coef2 * x_t

        if t[0] > 0:
            var = self.posterior_variance[t]
            while var.dim() < x_t.dim():
                var = var.unsqueeze(-1)
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(var) * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1
    
    def ddim_step(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        t_prev: torch.Tensor,
        noise_pred: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """DDIM 反向一步 - 更快的确定性采样"""
        # 获取 alpha 值
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev[0] >= 0 else torch.ones_like(alpha_t)
        
        while alpha_t.dim() < x_t.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_t_prev = alpha_t_prev.unsqueeze(-1)
        
        # 预测 x0
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # Clip 以提高稳定性
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # 计算 sigma
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        
        # 计算方向
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t_prev_minus_sigma_sq = torch.sqrt(
            torch.clamp(1 - alpha_t_prev - sigma_t ** 2, min=0)
        )
        
        # DDIM 更新
        x_t_prev = sqrt_alpha_t_prev * x_0_pred + sqrt_one_minus_alpha_t_prev_minus_sigma_sq * noise_pred
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma_t * noise
        
        return x_t_prev



# ============== 完整模型 ==============

class HybridDiffusionModel(nn.Module):
    """
    混合式条件扩散模型

    输入:
        - cond_coords: (B, T_cond, 4) 过去轨迹坐标 [lat, lon, vmax, pmin]
        - cond_era5: (B, T_cond, C, H, W) ERA5 局地场视频
        - cond_features: (B, T_cond, D) 环境特征
        - target_deltas: (B, T_future, 3) 目标增量 [Δlat, Δlon, Δvmax] (训练时)

    输出:
        - predicted_deltas: (B, T_future, 3) 预测增量
        - auxiliary_heatmap: (B, T_future, H, W) 辅助 heatmap (可选)
    """

    def __init__(
        self,
        coord_dim: int = 4,
        delta_dim: int = 3,
        era5_channels: int = 34,
        feature_dim: int = 64,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        t_cond: int = 12,
        t_future: int = 12,
        use_heatmap_head: bool = True
    ):
        super().__init__()
        self.t_cond = t_cond
        self.t_future = t_future
        self.delta_dim = delta_dim
        self.use_heatmap_head = use_heatmap_head

        # 编码器（针对 4090 增强）
        coord_embed_dim = 128   # 64 → 128
        era5_out_dim = 256      # 128 → 256
        feat_out_dim = 128      # 64 → 128

        self.coord_encoder = CoordEncoder(coord_dim, coord_embed_dim)
        self.era5_encoder = ERA5Encoder(era5_channels, era5_out_dim, base_ch=64)
        self.feat_encoder = FeatureEncoder(feature_dim, feat_out_dim)

        # 条件融合
        self.cond_fusion = ConditionFusion(
            coord_embed_dim, era5_out_dim, feat_out_dim, d_model
        )

        # 去噪网络
        self.denoiser = TransformerDenoiser(
            delta_dim=delta_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=d_model * 4,
            t_future=t_future
        )

        # DDPM 调度器
        self.scheduler = DDPMScheduler(
            num_steps=model_cfg.num_diffusion_steps,
            beta_start=model_cfg.beta_start,
            beta_end=model_cfg.beta_end,
            beta_schedule=model_cfg.beta_schedule
        )

        # 辅助 Heatmap Head
        if use_heatmap_head:
            self.heatmap_head = HeatmapHead(
                grid_h=data_cfg.grid_height,
                grid_w=data_cfg.grid_width,
                sigma=model_cfg.gaussian_sigma
            )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重以提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_conditions(
        self,
        cond_coords: torch.Tensor,
        cond_era5: torch.Tensor,
        cond_features: torch.Tensor
    ) -> torch.Tensor:
        """编码所有条件输入"""
        coord_emb = self.coord_encoder(cond_coords)
        era5_emb = self.era5_encoder(cond_era5)
        feat_emb = self.feat_encoder(cond_features)
        return self.cond_fusion(coord_emb, era5_emb, feat_emb)

    def forward(
        self,
        cond_coords: torch.Tensor,
        cond_era5: torch.Tensor,
        cond_features: torch.Tensor,
        target_deltas: torch.Tensor,
        target_coords: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """训练前向传播 - 增加物理约束和时序加权"""
        device = cond_coords.device
        B = cond_coords.shape[0]
        self.scheduler.to(device)

        # 编码条件
        cond_embed = self.encode_conditions(cond_coords, cond_era5, cond_features)

        # 随机时间步
        t = torch.randint(0, self.scheduler.num_steps, (B,), device=device)

        # 加噪
        noisy_deltas, noise = self.scheduler.add_noise(target_deltas, t)

        # 预测噪声
        noise_pred = self.denoiser(noisy_deltas, t, cond_embed)

        # === 时序加权 MSE 损失 ===
        T = target_deltas.shape[1]
        time_weights = torch.linspace(1.0, 1.5, T, device=device).view(1, T, 1)  # 降低权重范围
        
        # 加权 MSE
        mse_per_step = ((noise_pred - noise) ** 2) * time_weights
        mse_loss = mse_per_step.mean()
        
        # 检查 NaN
        if torch.isnan(mse_loss):
            mse_loss = torch.tensor(0.0, device=device, requires_grad=True)

        outputs = {
            'loss': mse_loss,
            'mse_loss': mse_loss,
        }

        # === 辅助 Heatmap 损失 ===
        if self.use_heatmap_head and target_coords is not None:
            # 从噪声预测恢复增量（添加数值稳定性）
            sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t].view(B, 1, 1)
            sqrt_alpha = torch.clamp(sqrt_alpha, min=1e-6)  # 避免除零
            sqrt_one_minus = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
            pred_deltas = (noisy_deltas - sqrt_one_minus * noise_pred) / sqrt_alpha
            pred_deltas = torch.clamp(pred_deltas, -10, 10)  # 限制范围

            # 计算预测坐标
            last_coord = cond_coords[:, -1:, :3]
            pred_coords = last_coord + torch.cumsum(pred_deltas, dim=1)
            pred_coords = torch.clamp(pred_coords, 0, 1)  # 归一化坐标范围

            # 归一化到网格坐标
            pred_heatmap = self.heatmap_head(
                pred_coords[:, :, 0] * (data_cfg.grid_height - 1),
                pred_coords[:, :, 1] * (data_cfg.grid_width - 1)
            )

            target_heatmap = self.heatmap_head(
                target_coords[:, :, 0] * (data_cfg.grid_height - 1),
                target_coords[:, :, 1] * (data_cfg.grid_width - 1)
            )

            heatmap_loss = F.mse_loss(pred_heatmap, target_heatmap)
            
            # 检查 NaN
            if torch.isnan(heatmap_loss):
                heatmap_loss = torch.tensor(0.0, device=device)
            
            outputs['heatmap_loss'] = heatmap_loss
            
            # === 物理约束损失 ===
            physics_loss = self._compute_physics_loss(pred_deltas, cond_coords)
            if torch.isnan(physics_loss):
                physics_loss = torch.tensor(0.0, device=device)
            outputs['physics_loss'] = physics_loss

            # 总损失
            outputs['loss'] = (
                mse_loss + 
                model_cfg.heatmap_loss_weight * heatmap_loss +
                0.01 * physics_loss  # 降低物理约束权重到 0.01
            )

        return outputs
    
    def _compute_physics_loss(
        self,
        pred_deltas: torch.Tensor,
        cond_coords: torch.Tensor,
        dt_hours: float = 0.5
    ) -> torch.Tensor:
        """计算物理约束损失 - 稳定版"""
        # 对于归一化的增量，值应该很小
        # 直接使用归一化空间计算，避免反归一化带来的数值问题
        
        # 1) 增量平滑性：相邻增量变化不应太大
        if pred_deltas.shape[1] > 1:
            delta_diff = torch.diff(pred_deltas, dim=1)
            smooth_loss = (delta_diff ** 2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=pred_deltas.device)
        
        # 2) 增量大小约束：归一化增量不应太大（每步变化有限）
        delta_magnitude = (pred_deltas ** 2).sum(dim=-1).sqrt()
        magnitude_loss = F.relu(delta_magnitude - 0.1).mean()  # 每步增量不超过 0.1
        
        # 总损失
        total = smooth_loss + magnitude_loss
        return total

    @torch.no_grad()
    def sample(
        self,
        cond_coords: torch.Tensor,
        cond_era5: torch.Tensor,
        cond_features: torch.Tensor,
        num_inference_steps: int = None,
        use_ddim: bool = True,
        eta: float = 0.0
    ) -> torch.Tensor:
        """采样生成 - 支持 DDIM 加速"""
        device = cond_coords.device
        B = cond_coords.shape[0]
        num_inference_steps = num_inference_steps or 50  # 默认使用 50 步 DDIM
        self.scheduler.to(device)

        # 编码条件
        cond_embed = self.encode_conditions(cond_coords, cond_era5, cond_features)

        # 初始化噪声
        x = torch.randn(B, self.t_future, self.delta_dim, device=device)

        if use_ddim:
            # DDIM 采样 - 更快更稳定
            # 创建时间步序列（均匀间隔）
            step_ratio = self.scheduler.num_steps // num_inference_steps
            timesteps = torch.arange(0, self.scheduler.num_steps, step_ratio, device=device)
            timesteps = torch.flip(timesteps, [0])  # 从大到小
            
            for i, t_val in enumerate(timesteps):
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                
                # 计算前一个时间步
                if i + 1 < len(timesteps):
                    t_prev = torch.full((B,), timesteps[i + 1], device=device, dtype=torch.long)
                else:
                    t_prev = torch.full((B,), 0, device=device, dtype=torch.long)
                
                noise_pred = self.denoiser(x, t, cond_embed)
                x = self.scheduler.ddim_step(x, t, t_prev, noise_pred, eta=eta)
        else:
            # 标准 DDPM 采样
            timesteps = torch.linspace(self.scheduler.num_steps - 1, 0, num_inference_steps, dtype=torch.long)
            
            for t_val in timesteps:
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                noise_pred = self.denoiser(x, t, cond_embed)
                x = self.scheduler.step(x, t, noise_pred)

        return x  # (B, T_future, delta_dim)

    def predict(
        self,
        cond_coords: torch.Tensor,
        cond_era5: torch.Tensor,
        cond_features: torch.Tensor,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """完整预测：返回绝对坐标"""
        all_deltas = []

        for _ in range(num_samples):
            deltas = self.sample(cond_coords, cond_era5, cond_features)
            all_deltas.append(deltas)

        # 平均增量
        mean_deltas = torch.stack(all_deltas).mean(dim=0)

        # 转换为绝对坐标
        last_coord = cond_coords[:, -1:, :3]  # [lat, lon, vmax]
        abs_coords = last_coord + torch.cumsum(mean_deltas, dim=1)

        result = {
            'predicted_deltas': mean_deltas,
            'predicted_lat': abs_coords[:, :, 0],
            'predicted_lon': abs_coords[:, :, 1],
            'predicted_vmax': abs_coords[:, :, 2],
            'all_samples': all_deltas if num_samples > 1 else None
        }

        # 生成辅助 heatmap
        if self.use_heatmap_head:
            result['auxiliary_heatmap'] = self.heatmap_head(
                abs_coords[:, :, 0] * (data_cfg.grid_height - 1),
                abs_coords[:, :, 1] * (data_cfg.grid_width - 1)
            )

        return result
