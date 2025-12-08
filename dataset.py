"""
PyTorch Dataset 模块：滑动窗口采样，train/val/test 划分
混合式条件扩散模型 - 台风路径预测
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import random

from config import model_cfg, train_cfg, data_cfg
from data_structures import StormSample, TrainingSample
from features import build_cond_features_for_storm


def normalize_coords(lat, lon, vmax, pmin=None, lat_range=(5, 40), lon_range=(100, 180)):
    """归一化坐标到 [0, 1] 范围"""
    lat_norm = (lat - lat_range[0]) / (lat_range[1] - lat_range[0])
    lon_norm = (lon - lon_range[0]) / (lon_range[1] - lon_range[0])
    vmax_norm = vmax / 100.0  # 假设最大风速不超过 100 m/s
    if pmin is not None:
        pmin_norm = (pmin - 870) / (1020 - 870)  # 气压范围 870-1020 hPa
        return lat_norm, lon_norm, vmax_norm, pmin_norm
    return lat_norm, lon_norm, vmax_norm


def compute_deltas(coords):
    """计算坐标增量"""
    # coords: (T, 3 or 4)
    deltas = np.diff(coords, axis=0)
    return deltas


class TyphoonHybridDataset(Dataset):
    """
    台风路径混合数据集
    输入：轨迹坐标 + ERA5 视频 + 环境特征
    输出：坐标增量
    """

    def __init__(
        self,
        storm_samples: List[StormSample],
        t_cond: int = None,
        t_future: int = None,
        stride: int = 1,
        use_sample_weights: bool = True,
        era5_channels: int = 34,
    ):
        self.storm_samples = storm_samples
        self.t_cond = t_cond or model_cfg.t_cond
        self.t_future = t_future or model_cfg.t_future
        self.stride = stride
        self.use_sample_weights = use_sample_weights
        self.era5_channels = era5_channels

        # 构建索引
        self.samples_index = self._build_samples_index()

    def _build_samples_index(self) -> List[Tuple[int, int]]:
        """构建滑动窗口索引"""
        index = []
        total_length = self.t_cond + self.t_future

        for storm_idx, sample in enumerate(self.storm_samples):
            T = len(sample)
            for start in range(0, T - total_length + 1, self.stride):
                index.append((storm_idx, start))

        return index

    def __len__(self) -> int:
        return len(self.samples_index)

    def _get_era5_video(self, sample: StormSample, start: int, end: int) -> np.ndarray:
        """获取 ERA5 视频数据"""
        T = end - start
        if sample.era5_array is not None:
            return sample.era5_array[start:end]
        elif sample.era5_dataset is not None:
            frames = []
            for t in range(start, end):
                frame = sample.get_era5_at_time(t)
                frames.append(frame)
            return np.stack(frames, axis=0)
        else:
            # 返回虚拟数据
            return np.zeros((T, self.era5_channels, data_cfg.grid_height, data_cfg.grid_width), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        storm_idx, start_idx = self.samples_index[idx]
        sample = self.storm_samples[storm_idx]

        # 计算索引范围
        cond_start = start_idx
        cond_end = start_idx + self.t_cond
        future_start = cond_end
        future_end = cond_end + self.t_future

        # === 条件输入 ===
        # 1. 轨迹坐标
        cond_lat = sample.track_lat[cond_start:cond_end]
        cond_lon = sample.track_lon[cond_start:cond_end]
        cond_vmax = sample.track_vmax[cond_start:cond_end]
        cond_pmin = sample.track_pmin[cond_start:cond_end] if sample.track_pmin is not None else np.zeros_like(cond_vmax)

        # 归一化
        lat_n, lon_n, vmax_n, pmin_n = normalize_coords(cond_lat, cond_lon, cond_vmax, cond_pmin)
        cond_coords = np.stack([lat_n, lon_n, vmax_n, pmin_n], axis=-1)  # (T_cond, 4)

        # 2. ERA5 视频
        cond_era5 = self._get_era5_video(sample, cond_start, cond_end)  # (T_cond, C, H, W)

        # 3. 环境特征
        all_features = build_cond_features_for_storm(sample)
        cond_features = all_features[cond_start:cond_end]  # (T_cond, D)

        # === 目标输出 ===
        # 未来坐标
        future_lat = sample.track_lat[future_start:future_end]
        future_lon = sample.track_lon[future_start:future_end]
        future_vmax = sample.track_vmax[future_start:future_end]
        future_pmin = sample.track_pmin[future_start:future_end] if sample.track_pmin is not None else np.zeros_like(future_vmax)

        # 归一化未来坐标
        f_lat_n, f_lon_n, f_vmax_n, f_pmin_n = normalize_coords(future_lat, future_lon, future_vmax, future_pmin)
        target_coords = np.stack([f_lat_n, f_lon_n, f_vmax_n, f_pmin_n], axis=-1)  # (T_future, 4)

        # 计算增量：相对于条件序列最后一个点
        last_cond = cond_coords[-1:, :3]  # (1, 3)
        future_3d = target_coords[:, :3]  # (T_future, 3)
        # 增量是相对于前一个点的变化
        target_deltas = np.diff(
            np.concatenate([last_cond, future_3d], axis=0), axis=0
        )  # (T_future, 3)

        # 样本权重
        sample_weight = 1.0
        if self.use_sample_weights and sample.is_real is not None:
            real_ratio = sample.is_real[future_start:future_end].mean()
            sample_weight = train_cfg.interp_sample_weight + \
                real_ratio * (train_cfg.real_sample_weight - train_cfg.interp_sample_weight)

        return {
            'cond_coords': torch.from_numpy(cond_coords).float(),
            'cond_era5': torch.from_numpy(cond_era5).float(),
            'cond_features': torch.from_numpy(cond_features).float(),
            'target_deltas': torch.from_numpy(target_deltas).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'sample_weight': torch.tensor(sample_weight).float(),
            'storm_id': sample.storm_id,
            # 保留原始坐标用于评估
            'target_lat_raw': torch.from_numpy(future_lat).float(),
            'target_lon_raw': torch.from_numpy(future_lon).float(),
        }


def split_storms_by_id(
    storm_samples: List[StormSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[StormSample], List[StormSample], List[StormSample]]:
    """按 storm_id 划分训练/验证/测试集

    确保每个集合至少有1个样本（当数据量足够时）
    """
    random.seed(seed)
    samples = storm_samples.copy()
    random.shuffle(samples)

    n = len(samples)

    # 特殊处理小数据集，确保每个集合至少有1个样本
    if n <= 3:
        # 数据太少，全部用于训练
        return samples, [], []
    elif n <= 6:
        # 小数据集：n-2个训练，1个验证，1个测试
        n_train = n - 2
        n_val = 1
    else:
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # 确保验证集和测试集至少各有1个
        if n_val == 0:
            n_val = 1
        if n_train + n_val >= n:
            n_train = n - n_val - 1

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    return train_samples, val_samples, test_samples


def split_storms_by_year(
    storm_samples: List[StormSample],
    train_years: List[int],
    val_years: List[int],
    test_years: List[int]
) -> Tuple[List[StormSample], List[StormSample], List[StormSample]]:
    """按年份划分训练/验证/测试集"""
    train_samples = [s for s in storm_samples if s.year in train_years]
    val_samples = [s for s in storm_samples if s.year in val_years]
    test_samples = [s for s in storm_samples if s.year in test_years]
    
    return train_samples, val_samples, test_samples


def create_dataloaders(
    storm_samples: List[StormSample],
    batch_size: int = None,
    split_by: str = 'storm_id',
    **split_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试 DataLoader"""
    batch_size = batch_size or train_cfg.batch_size

    # 划分数据
    if split_by == 'storm_id':
        train_s, val_s, test_s = split_storms_by_id(
            storm_samples,
            train_cfg.train_ratio,
            train_cfg.val_ratio
        )
    else:
        train_s, val_s, test_s = split_storms_by_year(storm_samples, **split_kwargs)

    # 创建 Dataset（使用新的混合数据集）
    train_ds = TyphoonHybridDataset(train_s, stride=1)
    val_ds = TyphoonHybridDataset(val_s, stride=model_cfg.t_future)
    test_ds = TyphoonHybridDataset(test_s, stride=model_cfg.t_future)

    # 创建 DataLoader（优化参数提高GPU利用率）
    num_workers = train_cfg.num_workers
    pin_memory = train_cfg.pin_memory

    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
