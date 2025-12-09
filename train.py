"""
训练脚本：训练混合式条件扩散模型
Hybrid Coordinate + Diffusion - 台风路径预测
"""
import os
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config import train_cfg, model_cfg, data_cfg
from model import HybridDiffusionModel
from dataset import TyphoonHybridDataset, create_dataloaders, split_storms_by_id
from data_processing import load_all_storms, load_tyc_storms


class Trainer:
    """训练器 - 增强版"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = None,
        learning_rate: float = None,
        num_epochs: int = None,
        checkpoint_dir: str = None
    ):
        self.device = device or train_cfg.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs or train_cfg.num_epochs
        self.checkpoint_dir = Path(checkpoint_dir or train_cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        learning_rate = learning_rate or train_cfg.learning_rate
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.999)  # 标准 AdamW 参数
        )

        # 学习率调度器 - 使用 Warmup + Cosine
        self.warmup_epochs = getattr(train_cfg, 'warmup_epochs', 5)
        self.scheduler = self._create_scheduler(learning_rate)

        # 日志
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        # 早停机制
        self.early_stopping = train_cfg.early_stopping
        self.patience = train_cfg.patience
        self.patience_counter = 0

        # 混合精度训练
        self.use_amp = train_cfg.use_amp and self.device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # EMA (Exponential Moving Average) 用于更稳定的预测
        self.use_ema = True
        self.ema_decay = 0.9999
        self.ema_model = None
        if self.use_ema:
            self._init_ema()
    
    def _create_scheduler(self, learning_rate: float):
        """创建带 Warmup 的 Cosine 学习率调度器"""
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # 线性 warmup
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine 衰减
                progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _init_ema(self):
        """初始化 EMA 模型"""
        import copy
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def _update_ema(self):
        """更新 EMA 模型参数"""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            # 移动数据到设备（non_blocking加速传输）
            cond_coords = batch['cond_coords'].to(self.device, non_blocking=True)
            cond_era5 = batch['cond_era5'].to(self.device, non_blocking=True)
            cond_features = batch['cond_features'].to(self.device, non_blocking=True)
            target_deltas = batch['target_deltas'].to(self.device, non_blocking=True)
            target_coords = batch['target_coords'].to(self.device, non_blocking=True)
            sample_weight = batch['sample_weight'].to(self.device, non_blocking=True)

            # 前向传播（混合精度）
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(
                    cond_coords, cond_era5, cond_features,
                    target_deltas, target_coords
                )
                # 加权损失
                loss = outputs['loss']
                if train_cfg.use_sample_weights:
                    loss = (loss * sample_weight).mean()

            # 反向传播（混合精度）
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # 更新 EMA
            if self.use_ema:
                self._update_ema()

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            postfix = {'loss': f"{loss.item():.4f}", 'mse': f"{outputs['mse_loss'].item():.4f}"}
            if 'heatmap_loss' in outputs:
                postfix['hm'] = f"{outputs['heatmap_loss'].item():.4f}"
            if 'physics_loss' in outputs:
                postfix['phy'] = f"{outputs['physics_loss'].item():.4f}"
            pbar.set_postfix(postfix)

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            cond_coords = batch['cond_coords'].to(self.device)
            cond_era5 = batch['cond_era5'].to(self.device)
            cond_features = batch['cond_features'].to(self.device)
            target_deltas = batch['target_deltas'].to(self.device)
            target_coords = batch['target_coords'].to(self.device)

            outputs = self.model(
                cond_coords, cond_era5, cond_features,
                target_deltas, target_coords
            )
            total_loss += outputs['loss'].item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点 - 保存最佳模型和 EMA 模型"""
        if not is_best:
            return  # 只保存最佳模型

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存 EMA 模型
        if self.use_ema and self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        # 只保存最佳模型
        torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
        print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint.get('epoch', 0)
    
    def train(self, resume_from: str = None):
        """完整训练循环"""
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed from epoch {start_epoch}")
        
        print(f"Training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        for epoch in range(start_epoch, self.num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step()

            # 打印日志
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # 保存检查点 - 如果验证集为空则使用训练损失
            compare_loss = val_loss if val_loss > 0 else train_loss
            is_best = compare_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = compare_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, is_best)

            # 早停检查
            if self.early_stopping and self.patience_counter >= self.patience:
                print(f"Early stopping triggered! No improvement for {self.patience} epochs.")
                break

        # 训练结束后保存loss图
        self.save_loss_plot()

        print("Training complete!")
        return self.train_losses, self.val_losses

    def save_config(self, config_dict: dict):
        """保存训练配置到JSON文件"""
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"Config saved to {config_path}")

    def save_loss_plot(self):
        """保存loss曲线图"""
        if not self.train_losses or not self.val_losses:
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)

        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)

        # 标记最佳点
        best_epoch = np.argmin(self.val_losses) + 1
        best_val = min(self.val_losses)
        plt.scatter([best_epoch], [best_val], c='green', s=100, zorder=5, label=f'Best (epoch {best_epoch})')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 保存图片
        plot_path = self.checkpoint_dir / 'loss_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to {plot_path}")


def evaluate_on_test(model, test_loader, device, use_ensemble: bool = True, num_samples: int = 5):
    """在测试集上评估模型性能 - 支持集合预测"""
    model.eval()

    if len(test_loader.dataset) == 0:
        print("  No test samples available!")
        return {}

    all_errors_km = []
    lat_range = (5, 40)
    lon_range = (100, 180)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            cond_coords = batch['cond_coords'].to(device)
            cond_era5 = batch['cond_era5'].to(device)
            cond_features = batch['cond_features'].to(device)
            target_deltas = batch['target_deltas'].to(device)
            target_lat = batch['target_lat_raw'].numpy()
            target_lon = batch['target_lon_raw'].numpy()

            # 集合预测：多次采样取平均
            if use_ensemble and num_samples > 1:
                all_pred_deltas = []
                for _ in range(num_samples):
                    pred_delta = model.sample(
                        cond_coords, cond_era5, cond_features,
                        num_inference_steps=50, use_ddim=True, eta=0.0
                    ).cpu().numpy()
                    all_pred_deltas.append(pred_delta)
                pred_delta = np.mean(all_pred_deltas, axis=0)
            else:
                pred_delta = model.sample(
                    cond_coords, cond_era5, cond_features,
                    num_inference_steps=50, use_ddim=True, eta=0.0
                ).cpu().numpy()

            # 反归一化
            last_cond = cond_coords[:, -1, :].cpu().numpy()
            last_lat_norm, last_lon_norm = last_cond[:, 0], last_cond[:, 1]

            pred_lat_delta = pred_delta[:, :, 0] * (lat_range[1] - lat_range[0])
            pred_lon_delta = pred_delta[:, :, 1] * (lon_range[1] - lon_range[0])

            last_lat = last_lat_norm * (lat_range[1] - lat_range[0]) + lat_range[0]
            last_lon = last_lon_norm * (lon_range[1] - lon_range[0]) + lon_range[0]

            pred_lat = last_lat[:, None] + np.cumsum(pred_lat_delta, axis=1)
            pred_lon = last_lon[:, None] + np.cumsum(pred_lon_delta, axis=1)

            # 计算误差(km)
            lat_err = (pred_lat - target_lat) * 111
            lon_err = (pred_lon - target_lon) * 111 * np.cos(np.radians(target_lat))
            dist_err = np.sqrt(lat_err**2 + lon_err**2)
            all_errors_km.append(dist_err)

    all_errors = np.concatenate(all_errors_km, axis=0)

    # 打印结果
    print(f"  Test samples: {all_errors.shape[0]}, Timesteps: {all_errors.shape[1]}")
    print(f"  Mean Error: {all_errors.mean():.2f} km")
    print(f"  Error by forecast hour:")

    results = {
        "num_samples": int(all_errors.shape[0]),
        "mean_error_km": float(all_errors.mean()),
        "std_error_km": float(all_errors.std()),
        "error_by_hour": {}
    }

    for t in range(all_errors.shape[1]):
        hours = (t + 1) * 0.5
        mean_err = all_errors[:, t].mean()
        std_err = all_errors[:, t].std()
        print(f"    +{hours:.1f}h: {mean_err:.2f} ± {std_err:.2f} km")
        results["error_by_hour"][f"{hours}h"] = {
            "mean_km": float(mean_err),
            "std_km": float(std_err)
        }

    return results


def main():
    """主函数：加载数据并训练模型"""
    print("=" * 60)
    print("Hybrid Diffusion Typhoon Track Prediction - Training")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] Loading data...")
    try:
        # 使用TYC格式加载器
        storm_samples = load_tyc_storms(
            csv_path=data_cfg.csv_path,
            era5_base_dir=data_cfg.era5_dir
        )
    except Exception as e:
        print(f"Error loading TYC data: {e}")
        print("Trying standard loader...")
        try:
            storm_samples = load_all_storms()
        except FileNotFoundError as e2:
            print(f"Data files not found: {e2}")
            print("Creating dummy data for testing...")
            storm_samples = create_dummy_data()

    if len(storm_samples) == 0:
        print("No data available. Please check your data paths in config.py")
        return

    # 2. 创建数据加载器
    print("\n[2] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(storm_samples)

    # 3. 创建模型
    print("\n[3] Creating model...")
    # 获取条件特征维度
    sample_batch = next(iter(train_loader))
    cond_feature_dim = sample_batch['cond_features'].shape[-1]
    era5_channels = sample_batch['cond_era5'].shape[2]
    print(f"Condition feature dimension: {cond_feature_dim}")
    print(f"ERA5 channels: {era5_channels}")

    model = HybridDiffusionModel(
        coord_dim=model_cfg.coord_dim,
        delta_dim=model_cfg.delta_dim,
        era5_channels=era5_channels,
        feature_dim=cond_feature_dim,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        t_cond=model_cfg.t_cond,
        t_future=model_cfg.t_future,
        use_heatmap_head=model_cfg.use_heatmap_head
    )

    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # 4. 训练
    print("\n[4] Starting training...")
    trainer = Trainer(model, train_loader, val_loader)

    # 保存训练配置
    config = {
        "model": {
            "coord_dim": model_cfg.coord_dim,
            "delta_dim": model_cfg.delta_dim,
            "era5_channels": era5_channels,
            "feature_dim": cond_feature_dim,
            "transformer_dim": model_cfg.transformer_dim,
            "transformer_heads": model_cfg.transformer_heads,
            "transformer_layers": model_cfg.transformer_layers,
            "t_cond": model_cfg.t_cond,
            "t_future": model_cfg.t_future,
            "use_heatmap_head": model_cfg.use_heatmap_head,
            "num_params": num_params
        },
        "training": {
            "batch_size": train_cfg.batch_size,
            "learning_rate": train_cfg.learning_rate,
            "num_epochs": train_cfg.num_epochs,
            "device": train_cfg.device
        },
        "data": {
            "csv_path": data_cfg.csv_path,
            "era5_dir": "E:/TYC",
            "num_storms": len(storm_samples),
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset)
        }
    }
    trainer.save_config(config)

    # 开始训练
    trainer.train()

    # 5. 测试评估
    print("\n[5] Evaluating on test set...")
    # 使用 EMA 模型进行评估（如果有）
    eval_model = trainer.ema_model if (trainer.use_ema and trainer.ema_model is not None) else model
    test_results = evaluate_on_test(eval_model, test_loader, trainer.device, use_ensemble=True, num_samples=10)

    # 保存测试结果到config
    config["test_results"] = test_results
    trainer.save_config(config)

    print("\nDone!")


def create_dummy_data():
    """创建测试用的虚拟数据"""
    from data_structures import StormSample
    import numpy as np

    dummy_samples = []

    for i in range(5):  # 5 个虚拟台风
        T = 50  # 50 个时间步
        storm_id = f"STORM_{i:04d}"
        times = np.array([np.datetime64('2020-01-01') + np.timedelta64(j * 30, 'm') for j in range(T)])

        # 模拟台风轨迹（从西向东北移动）
        track_lat = 15 + i + np.linspace(0, 15, T) + np.random.randn(T) * 0.5
        track_lon = 120 + np.linspace(0, 20, T) + np.random.randn(T) * 0.5
        track_vmax = 30 + 20 * np.sin(np.linspace(0, np.pi, T)) + np.random.randn(T) * 2

        # 创建虚拟网格
        lat_grid = np.linspace(track_lat.min() - 5, track_lat.max() + 5, data_cfg.grid_height)
        lon_grid = np.linspace(track_lon.min() - 5, track_lon.max() + 5, data_cfg.grid_width)

        sample = StormSample(
            storm_id=storm_id,
            times=times,
            track_lat=track_lat.astype(np.float32),
            track_lon=track_lon.astype(np.float32),
            track_vmax=track_vmax.astype(np.float32),
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            is_real=np.array([j % 6 == 0 for j in range(T)]),  # 每6个时间步为真实数据
            year=2020 + i % 3
        )
        dummy_samples.append(sample)

    print(f"Created {len(dummy_samples)} dummy storm samples")
    return dummy_samples


if __name__ == "__main__":
    main()

