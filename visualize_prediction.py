"""
可视化台风路径预测结果
同时显示真实路径和预测路径
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from torch.utils.data import DataLoader

from config import model_cfg
from model import HybridDiffusionModel
from dataset import TyphoonHybridDataset, split_storms_by_id
from data_processing import load_tyc_storms


def load_model_and_data():
    """加载模型和测试数据"""
    print('Loading data...')
    storm_samples = load_tyc_storms(era5_base_dir='E:/TYC')
    train_s, val_s, test_s = split_storms_by_id(storm_samples, 0.7, 0.15, seed=42)
    
    print(f'Test storms: {len(test_s)}')
    for s in test_s:
        print(f'  - {s.storm_id}: {len(s.times)} timesteps')
    
    test_ds = TyphoonHybridDataset(test_s, stride=model_cfg.t_future)
    print(f'Test samples: {len(test_ds)}')
    
    # 加载模型
    print('\nLoading best model...')
    checkpoint = torch.load('checkpoints/best.pt', map_location='cuda', weights_only=False)
    sample = test_ds[0]
    era5_channels = sample['cond_era5'].shape[1]
    feature_dim = sample['cond_features'].shape[-1]
    
    model = HybridDiffusionModel(
        coord_dim=model_cfg.coord_dim,
        delta_dim=model_cfg.delta_dim,
        era5_channels=era5_channels,
        feature_dim=feature_dim,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        t_cond=model_cfg.t_cond,
        t_future=model_cfg.t_future,
        use_heatmap_head=model_cfg.use_heatmap_head
    ).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Loaded from epoch {checkpoint["epoch"]+1}')
    
    return model, test_ds, test_s


def predict_and_visualize(model, test_ds, test_storms, output_dir='predictions'):
    """预测并可视化结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    lat_range, lon_range = (5, 40), (100, 180)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    all_results = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            cond_coords = batch['cond_coords'].cuda()
            cond_era5 = batch['cond_era5'].cuda()
            cond_features = batch['cond_features'].cuda()
            target_lat = batch['target_lat_raw'].numpy()[0]
            target_lon = batch['target_lon_raw'].numpy()[0]
            storm_id = batch['storm_id'][0]
            
            # 条件轨迹（历史）
            cond_coords_np = cond_coords.cpu().numpy()[0]
            cond_lat = cond_coords_np[:, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
            cond_lon = cond_coords_np[:, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]
            
            # 预测
            pred_delta = model.sample(cond_coords, cond_era5, cond_features).cpu().numpy()[0]
            
            last_lat_norm = cond_coords_np[-1, 0]
            last_lon_norm = cond_coords_np[-1, 1]
            
            pred_lat_delta = pred_delta[:, 0] * (lat_range[1] - lat_range[0])
            pred_lon_delta = pred_delta[:, 1] * (lon_range[1] - lon_range[0])
            
            last_lat = last_lat_norm * (lat_range[1] - lat_range[0]) + lat_range[0]
            last_lon = last_lon_norm * (lon_range[1] - lon_range[0]) + lon_range[0]
            
            pred_lat = last_lat + np.cumsum(pred_lat_delta)
            pred_lon = last_lon + np.cumsum(pred_lon_delta)
            
            # 计算误差
            lat_err = (pred_lat - target_lat) * 111
            lon_err = (pred_lon - target_lon) * 111 * np.cos(np.radians(target_lat))
            dist_err = np.sqrt(lat_err**2 + lon_err**2)
            
            all_results.append({
                'storm_id': storm_id,
                'sample_idx': idx,
                'cond_lat': cond_lat,
                'cond_lon': cond_lon,
                'target_lat': target_lat,
                'target_lon': target_lon,
                'pred_lat': pred_lat,
                'pred_lon': pred_lon,
                'error_km': dist_err
            })
    
    # 只绘制第一个样本
    plot_single_prediction(all_results[0], output_dir)
    
    return all_results


def plot_single_prediction(result, output_dir):
    """绘制单个预测结果"""
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 固定地图范围: 100-180E, 0-60N
    extent = [100, 180, 0, 60]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # 添加地图特征
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # 绘制历史轨迹（条件输入）
    ax.plot(result['cond_lon'], result['cond_lat'], 'b-', linewidth=2.5,
            transform=ccrs.PlateCarree(), label='History (Input)', zorder=5)
    ax.scatter(result['cond_lon'], result['cond_lat'], c='blue', s=30,
               transform=ccrs.PlateCarree(), zorder=6)
    
    # 绘制真实未来轨迹
    full_true_lon = np.concatenate([[result['cond_lon'][-1]], result['target_lon']])
    full_true_lat = np.concatenate([[result['cond_lat'][-1]], result['target_lat']])
    ax.plot(full_true_lon, full_true_lat, 'g-', linewidth=2.5,
            transform=ccrs.PlateCarree(), label='Ground Truth', zorder=7)
    ax.scatter(result['target_lon'], result['target_lat'], c='green', s=30,
               transform=ccrs.PlateCarree(), zorder=8)
    
    # 绘制预测轨迹
    full_pred_lon = np.concatenate([[result['cond_lon'][-1]], result['pred_lon']])
    full_pred_lat = np.concatenate([[result['cond_lat'][-1]], result['pred_lat']])
    ax.plot(full_pred_lon, full_pred_lat, 'r--', linewidth=2.5,
            transform=ccrs.PlateCarree(), label='Prediction', zorder=9)
    ax.scatter(result['pred_lon'], result['pred_lat'], c='red', s=30, marker='x',
               transform=ccrs.PlateCarree(), zorder=10)
    
    # 标记起始点
    ax.scatter(result['cond_lon'][-1], result['cond_lat'][-1], c='black', s=100, marker='*',
               transform=ccrs.PlateCarree(), label='Forecast Start', zorder=11)
    
    # 标题和图例
    mean_err = result['error_km'].mean()
    final_err = result['error_km'][-1]
    plt.title(f"Typhoon {result['storm_id']} - Sample {result['sample_idx']}\n"
              f"Mean Error: {mean_err:.1f} km | 6h Error: {final_err:.1f} km",
              fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # 保存
    save_path = output_dir / f"pred_{result['storm_id']}_sample{result['sample_idx']:02d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_all_predictions(all_results, output_dir):
    """绘制所有预测结果的汇总图"""
    n_samples = len(all_results)
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    if n_samples == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(all_results):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # 计算地图范围
        all_lats = np.concatenate([result['cond_lat'], result['target_lat'], result['pred_lat']])
        all_lons = np.concatenate([result['cond_lon'], result['target_lon'], result['pred_lon']])
        
        lat_margin = max(2, (all_lats.max() - all_lats.min()) * 0.15)
        lon_margin = max(2, (all_lons.max() - all_lons.min()) * 0.15)
        
        extent = [all_lons.min() - lon_margin, all_lons.max() + lon_margin,
                  all_lats.min() - lat_margin, all_lats.max() + lat_margin]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        
        # 绘制轨迹
        ax.plot(result['cond_lon'], result['cond_lat'], 'b-', linewidth=1.5, transform=ccrs.PlateCarree())
        
        full_true_lon = np.concatenate([[result['cond_lon'][-1]], result['target_lon']])
        full_true_lat = np.concatenate([[result['cond_lat'][-1]], result['target_lat']])
        ax.plot(full_true_lon, full_true_lat, 'g-', linewidth=1.5, transform=ccrs.PlateCarree())
        
        full_pred_lon = np.concatenate([[result['cond_lon'][-1]], result['pred_lon']])
        full_pred_lat = np.concatenate([[result['cond_lat'][-1]], result['pred_lat']])
        ax.plot(full_pred_lon, full_pred_lat, 'r--', linewidth=1.5, transform=ccrs.PlateCarree())
        
        ax.scatter(result['cond_lon'][-1], result['cond_lat'][-1], c='black', s=50, marker='*',
                   transform=ccrs.PlateCarree())
        
        mean_err = result['error_km'].mean()
        ax.set_title(f"Sample {idx}: {mean_err:.0f} km", fontsize=10)
    
    # 隐藏空白子图
    for idx in range(n_samples, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='History'),
        Line2D([0], [0], color='green', linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Prediction'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12)
    
    plt.suptitle('Typhoon Track Prediction Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = output_dir / 'all_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def main():
    print('='*60)
    print('Typhoon Track Prediction Visualization')
    print('='*60)
    
    model, test_ds, test_storms = load_model_and_data()
    results = predict_and_visualize(model, test_ds, test_storms)
    
    # 打印误差统计
    print('\n' + '='*60)
    print('Error Statistics (km)')
    print('='*60)
    all_errors = np.array([r['error_km'] for r in results])
    print(f'Overall Mean: {all_errors.mean():.1f} km')
    print(f'Overall Std: {all_errors.std():.1f} km')
    print('\nBy forecast hour:')
    for t in range(all_errors.shape[1]):
        hours = (t + 1) * 0.5
        print(f'  +{hours:.1f}h: {all_errors[:, t].mean():.1f} ± {all_errors[:, t].std():.1f} km')
    
    print(f'\nVisualization saved to: predictions/')


if __name__ == '__main__':
    main()

