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
    """轨迹坐标编码器"""
    def __init__(self, coord_dim: int = 4, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


class ERA5Encoder(nn.Module):
    """ERA5 视频编码器"""
    def __init__(self, in_channels: int = 34, out_dim: int = 128, base_ch: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(base_ch * 4, out_dim)

    def forward(self, era5: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = era5.shape
        x = era5.view(B * T, C, H, W)
        x = self.conv(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x.view(B, T, -1)


class FeatureEncoder(nn.Module):
    """环境特征编码器"""
    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class ConditionFusion(nn.Module):
    """融合所有条件输入"""
    def __init__(self, coord_dim: int, era5_dim: int, feat_dim: int, out_dim: int):
        super().__init__()
        total_dim = coord_dim + era5_dim + feat_dim
        self.proj = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, coord_emb, era5_emb, feat_emb) -> torch.Tensor:
        x = torch.cat([coord_emb, era5_emb, feat_emb], dim=-1)
        return self.proj(x)


# ============== 去噪网络（Denoiser） ==============

class TransformerDenoiser(nn.Module):
    """
    Transformer-based 去噪网络
    输入: 噪声增量序列 (B, T_future, delta_dim) + 条件嵌入 (B, T_cond, d_model)
    输出: 预测噪声 (B, T_future, delta_dim)
    """
    def __init__(
        self,
        delta_dim: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        t_future: int = 12
    ):
        super().__init__()
        self.d_model = d_model
        self.t_future = t_future

        # 增量序列嵌入
        self.delta_embed = nn.Linear(delta_dim, d_model)

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, delta_dim)
        )

    def forward(
        self,
        noisy_deltas: torch.Tensor,  # (B, T_future, delta_dim)
        t: torch.Tensor,              # (B,) diffusion timestep
        cond_embed: torch.Tensor      # (B, T_cond, d_model)
    ) -> torch.Tensor:
        B = noisy_deltas.shape[0]

        # 嵌入噪声增量
        x = self.delta_embed(noisy_deltas)  # (B, T_future, d_model)
        x = self.pos_enc(x)

        # 注入时间步信息
        t_emb = self.time_embed(t)  # (B, d_model)
        x = x + t_emb.unsqueeze(1)

        # 拼接条件（cross-attention 的简化版：直接拼接）
        # cond_embed: (B, T_cond, d_model)
        x = torch.cat([cond_embed, x], dim=1)  # (B, T_cond + T_future, d_model)

        # Transformer
        x = self.transformer(x)

        # 只取后 T_future 个位置的输出
        x = x[:, -self.t_future:, :]  # (B, T_future, d_model)

        # 输出投影
        return self.out_proj(x)  # (B, T_future, delta_dim)


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
    """DDPM 噪声调度器"""

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        self.num_steps = num_steps

        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif beta_schedule == "cosine":
            steps = num_steps + 1
            s = 0.008
            x = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

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

        # 编码器
        coord_embed_dim = 64
        era5_out_dim = 128
        feat_out_dim = 64

        self.coord_encoder = CoordEncoder(coord_dim, coord_embed_dim)
        self.era5_encoder = ERA5Encoder(era5_channels, era5_out_dim)
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
        """训练前向传播"""
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

        # 损失
        mse_loss = F.mse_loss(noise_pred, noise)

        outputs = {
            'loss': mse_loss,
            'mse_loss': mse_loss,
        }

        # 辅助 Heatmap 损失
        if self.use_heatmap_head and target_coords is not None:
            # 从噪声预测恢复增量
            sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t].view(B, 1, 1)
            sqrt_one_minus = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
            pred_deltas = (noisy_deltas - sqrt_one_minus * noise_pred) / sqrt_alpha

            # 计算预测坐标
            last_coord = cond_coords[:, -1:, :3]  # (B, 1, 3) [lat, lon, vmax]
            pred_coords = last_coord + torch.cumsum(pred_deltas, dim=1)

            # 归一化到网格坐标
            # 简化处理：假设坐标已归一化
            pred_heatmap = self.heatmap_head(
                pred_coords[:, :, 0] * (data_cfg.grid_height - 1),
                pred_coords[:, :, 1] * (data_cfg.grid_width - 1)
            )

            target_heatmap = self.heatmap_head(
                target_coords[:, :, 0] * (data_cfg.grid_height - 1),
                target_coords[:, :, 1] * (data_cfg.grid_width - 1)
            )

            heatmap_loss = F.mse_loss(pred_heatmap, target_heatmap)
            outputs['heatmap_loss'] = heatmap_loss
            outputs['loss'] = mse_loss + model_cfg.heatmap_loss_weight * heatmap_loss

        return outputs

    @torch.no_grad()
    def sample(
        self,
        cond_coords: torch.Tensor,
        cond_era5: torch.Tensor,
        cond_features: torch.Tensor,
        num_inference_steps: int = None
    ) -> torch.Tensor:
        """采样生成"""
        device = cond_coords.device
        B = cond_coords.shape[0]
        num_inference_steps = num_inference_steps or self.scheduler.num_steps
        self.scheduler.to(device)

        # 编码条件
        cond_embed = self.encode_conditions(cond_coords, cond_era5, cond_features)

        # 初始化噪声
        x = torch.randn(B, self.t_future, self.delta_dim, device=device)

        # 反向扩散
        timesteps = torch.linspace(num_inference_steps - 1, 0, num_inference_steps, dtype=torch.long)

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
