# 任务：台风轨迹预测模型优化

## 一、项目背景

我正在开发一个基于深度学习的台风轨迹预测系统，需要你帮我按照以下架构和改进方向重构代码。

### 当前数据情况

- **输入数据**：
  - 台风轨迹 CSV（字段：storm_id, time, lat, lon, vmax, pmin）
  - ERA5 再分析场（多压力层 3D 变量 + 2D 变量）
- **预处理**：
  - 条件序列：t_cond=12（过去 6h，0.5h/步）
  - 预测目标：t_future=12（未来 6h，0.5h/步）
  - 归一化：lat/lon 映射 [0,1]，vmax/100，pmin 映射 870-1020
- **数据划分**：按 storm_id 划分 7/1.5/1.5（训练/验证/测试）

### 当前问题

使用 Hybrid Diffusion（DDPM 1000 步 + Transformer Denoiser）预测准确率很差，需要系统性改进。

---

## 二、改进目标与架构设计

### 核心设计原则

1. **不使用 Video2Video**：ERA5 仅作为条件编码，输出仍为轨迹 delta
2. **渐进式实验**：先验证基线，再逐步添加复杂度
3. **物理约束**：加入台风运动的物理先验
4. **高效编码**：用 3D 时空编码替代简单逐帧处理

### 目标架构概览

```

输入层：
├── 历史轨迹 (12, 5) → CoordEncoder (MLP)
├── ERA5 序列 (12, C, H, W) → ImprovedERA5Encoder (3D Conv + Temporal Attention)
└── 条件特征 (12, F) → FeatureEncoder (MLP)
↓
融合层：ConditionFusion → condition_embedding
↓
预测层（可切换）：
├── [基线] DeterministicPredictor → 直接输出 (12, 3)
└── [进阶] SimplifiedDiffusion (DDIM 100步) → 采样输出 (12, 3)
↓
损失函数：MSE + physics_loss + (可选) heatmap_loss
↓
输出：Δlat, Δlon, Δvmax (12步)

```

---

## 三、需要实现的模块

### 模块 1：改进的 ERA5 编码器

```python
class ImprovedERA5Encoder(nn.Module):
    """
    用 3D 卷积 + 时序注意力编码 ERA5 序列

    输入: (B, T, C, H, W) - T个时间步的ERA5场
    输出: (B, hidden_dim) - 条件嵌入

    要求：
    1. 使用 3D 卷积捕捉时空关系
    2. 可选：添加以台风位置为中心的空间注意力
    3. 使用 GroupNorm 而非 BatchNorm（小批量更稳定）
    4. 最终输出固定维度的嵌入向量
    """
    pass
```

### 模块 2：增强的条件特征

```python
# 需要计算的条件特征列表
condition_features = {
    # 时间特征
    'day_of_year_sin': '年内日正弦编码',
    'day_of_year_cos': '年内日余弦编码',

    # 运动学特征（从历史轨迹计算）
    'speed': '移动速度 (km/h)',
    'direction': '移动方向 (rad)',
    'acceleration': '加速度',
    'delta_lat': '纬度变化',
    'delta_lon': '经度变化',
    'delta_vmax': '强度变化',

    # ERA5 局地统计（台风中心附近）
    'wind_shear': '风切变 (200-850hPa)',
    'vorticity_850': '850hPa 涡度',
    'divergence_200': '200hPa 辐散',
    'msl': '海平面气压',
    'tcwv': '整层可降水量',
    'sst': '海表温度',

    # 引导气流（关键新增）
    'steering_u': '引导气流 u 分量 (加权平均 250/500/700hPa)',
    'steering_v': '引导气流 v 分量',

    # 标记
    'is_real_point': '是否为真实观测点（非插值）',
}
```

### 模块 3：物理约束损失

```python
def physics_loss(pred_deltas, cond_coords, cond_features):
    """
    台风运动的物理约束

    pred_deltas: (B, 12, 3) - 预测的 Δlat, Δlon, Δvmax
    cond_coords: (B, 12, 2) - 历史 lat, lon
    cond_features: (B, 12, F) - 历史特征（含速度、方向）

    约束项：
    1. 速度约束：台风移速通常 < 100 km/h，极少 > 150 km/h
    2. 平滑性约束：加速度不应突变
    3. 方向约束：转向通常平滑，急转弯罕见
    4. 强度约束：vmax 变化有物理上限（快速增强 ~30kt/24h）

    返回：标量损失值
    """
    pass
```

### 模块 4：双模式预测器

```python
class TrajectoryPredictor(nn.Module):
    """
    支持两种模式：
    1. deterministic: 直接回归，用于基线实验
    2. diffusion: 简化的 DDIM 扩散，用于概率预测

    通过 config 切换模式
    """

    def __init__(self, config):
        self.mode = config.prediction_mode  # 'deterministic' or 'diffusion'

        if self.mode == 'deterministic':
            self.predictor = DeterministicHead(...)
        else:
            self.predictor = SimplifiedDiffusion(
                num_steps=100,  # 不是1000
                schedule='cosine',
                prediction_type='x0',  # 直接预测目标，不是噪声
            )
```

### 模块 5：简化的扩散模块（如果使用）

```python
class SimplifiedDiffusion(nn.Module):
    """
    简化版扩散模型

    关键改进：
    1. 步数：100（不是1000）
    2. 调度：cosine（不是linear）
    3. 预测目标：直接预测 x0（不是噪声 epsilon）
    4. 采样器：DDIM（确定性，可进一步减少到 20-50 步）

    注意：输出维度很小（12×3=36），不需要复杂的 U-Net
    """
    pass
```

---

## 四、训练配置

```python
config = {
    # 数据
    'csv_path': 'path/to/typhoon_tracks.csv',
    'era5_dir': 'path/to/era5/',
    't_cond': 12,
    't_future': 12,
    'batch_size': 32,

    # 模型
    'hidden_dim': 256,
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 4,
    'prediction_mode': 'deterministic',  # 先用这个做基线

    # 扩散（如果使用）
    'diffusion_steps': 100,
    'diffusion_schedule': 'cosine',
    'diffusion_prediction_type': 'x0',
    'ddim_sampling_steps': 50,

    # 损失权重
    'mse_weight': 1.0,
    'physics_weight': 0.1,
    'heatmap_weight': 0.1,  # 可选

    # 优化
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 15,
    'grad_clip': 1.0,

    # 课程学习（可选）
    'use_curriculum': True,
    'curriculum_schedule': {
        0: 4,    # epoch 0-9: 预测 4 步
        10: 8,   # epoch 10-29: 预测 8 步
        30: 12,  # epoch 30+: 预测 12 步
    },
}
```

---

## 五、实验计划

请按以下顺序实现和验证：

### 阶段 1：数据验证

- [ ] 确认归一化/反归一化正确
- [ ] 可视化几个样本的 条件轨迹 → 目标轨迹 连续性
- [ ] 确认 ERA5 时间戳与轨迹点对齐
- [ ] 用真值作为预测，验证评估代码（误差应为 0）

### 阶段 2：基线模型

- [ ] 实现 ImprovedERA5Encoder（3D Conv）
- [ ] 实现 DeterministicPredictor（纯 Transformer + MSE）
- [ ] 训练并记录各步 MAE/RMSE (km)
- [ ] 这是后续对比的基准

### 阶段 3：添加物理约束

- [ ] 实现 physics_loss
- [ ] 加入训练（权重 0.1）
- [ ] 对比阶段 2，观察轨迹平滑度和误差变化

### 阶段 4：特征增强

- [ ] 添加引导气流（steering flow）计算
- [ ] 添加 SST、200hPa 辐散等变量
- [ ] 对比阶段 3

### 阶段 5：扩散模型（可选）

- [ ] 实现 SimplifiedDiffusion（DDIM 100 步，predict x0）
- [ ] 对比确定性基线
- [ ] 如果提升不明显，保持使用确定性模型

---

## 六、评估指标

```python
def evaluate(predictions, targets):
    """
    predictions: (N, 12, 3) - 预测的 Δlat, Δlon, Δvmax
    targets: (N, 12, 3) - 真实的 Δlat, Δlon, Δvmax

    返回：
    - 按步（1-12）的 MAE 和 RMSE（单位：km）
    - 总体距离误差
    - 强度误差（vmax）
    - 可选：转向点检测准确率
    """
    pass
```

需要输出格式：

```
Step |  MAE (km)  | RMSE (km) | vmax MAE (kt)
-----|------------|-----------|---------------
  1  |    15.2    |   20.1    |     2.3
  2  |    28.7    |   35.4    |     3.8
 ... |    ...     |   ...     |     ...
 12  |   125.3    |  158.2    |    12.5
-----|------------|-----------|---------------
Avg  |    68.4    |   89.6    |     7.2
```

---

## 七、代码结构

请按以下结构组织代码：

```
typhoon_prediction/
├── config.py              # 所有配置
├── data/
│   ├── dataset.py         # TyphoonDataset
│   ├── preprocessing.py   # 归一化、特征计算
│   └── era5_utils.py      # ERA5 加载、引导气流计算
├── models/
│   ├── encoders.py        # CoordEncoder, ImprovedERA5Encoder, FeatureEncoder
│   ├── fusion.py          # ConditionFusion
│   ├── predictors.py      # DeterministicPredictor, SimplifiedDiffusion
│   ├── losses.py          # physics_loss, heatmap_loss
│   └── model.py           # 主模型（组合以上模块）
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── visualize.py           # 可视化脚本
└── utils/
    ├── metrics.py         # MAE, RMSE, 距离计算
    └── callbacks.py       # 早停、学习率调度
```

---

## 八、额外要求

1. **代码规范**：使用 type hints，添加 docstring
2. **可复现性**：设置随机种子，记录所有超参数
3. **日志**：使用 logging 或 wandb 记录训练过程
4. **检查点**：保存最佳模型和最近模型
5. **可视化**：每个 epoch 生成几个样本的轨迹对比图

---

现在请按照阶段 2（基线模型）开始实现，先给我完整的模型代码。

```

```
