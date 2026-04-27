# 道路事件检测系统 (Road Event Detection with UniTS)

基于监控视频的端到端道路事件检测系统。核心流程：

**视频 → YOLO+ByteTrack车辆跟踪 → 时序特征工程 → UniTS/Transformer分类 → 事件类型**

支持检测的事件类型：正常通行、拥堵、事故、违法停车、逆行等。

---

## 项目结构

```
vehicle_tracker/
├── configs/
│   └── default.yaml              # 全局配置（模型、特征、训练参数）
├── src/
│   ├── video_tracker.py          # YOLOv8 + ByteTrack 车辆跟踪
│   ├── feature_engineering.py    # 时序特征提取（12维群体特征）
│   ├── dataset.py                # PyTorch Dataset
│   ├── train.py                  # 训练逻辑
│   ├── inference.py              # 端到端推理
│   ├── models/
│   │   ├── simple_tsc.py         # 简化版 Transformer 分类器
│   │   └── units_wrapper.py      # UniTS 官方模型适配器
│   └── utils.py                  # 工具函数
├── scripts/
│   ├── 00_prepare_annotations.py # 生成标注模板
│   ├── 01_extract_trajectories.py# 从视频提取轨迹
│   ├── 02_build_features.py      # 构建时序特征数据集
│   ├── 03_train.py               # 训练分类模型
│   └── 04_inference.py           # 推理/检测
├── data/
│   ├── raw_videos/               # 原始监控视频
│   ├── trajectories/             # 跟踪结果（JSON）
│   ├── features/                 # 时序特征（NPY）
│   └── annotations.json          # 标注文件（视频名 -> 类别索引）
├── checkpoints/                  # 模型权重
├── outputs/                      # 推理结果
└── requirements.txt
```

---

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

> 首次运行 YOLO 时会自动下载 `yolov8n.pt` 权重。

### 2. 准备数据

将你的监控视频放到 `data/raw_videos/` 目录下。

### 3. 完整流程（4步走）

#### Step 1: 提取车辆轨迹

```bash
python scripts/01_extract_trajectories.py \
    --video_dir ./data/raw_videos \
    --save_dir ./data/trajectories
```

输出：每帧的车辆ID、位置、类别等信息，保存为 `data/trajectories/{视频名}.json`。

#### Step 2-A: 生成标注模板（如果没有标注）

```bash
python scripts/00_prepare_annotations.py \
    --traj_dir ./data/trajectories \
    --output ./data/annotations.json
```

编辑 `data/annotations.json`，将每个视频标注为对应的事件类别索引：

```json
{
  "video_001": 1,    // 1 = congestion 拥堵
  "video_002": 0,    // 0 = normal 正常
  "video_003": 2     // 2 = accident 事故
}
```

类别定义见 `configs/default.yaml` 中 `events.classes`。

#### Step 2-B: 构建时序特征数据集

```bash
python scripts/02_build_features.py \
    --traj_dir ./data/trajectories \
    --save_dir ./data/features \
    --annotations ./data/annotations.json \
    --split
```

输出：`train_X.npy`, `train_y.npy`, `val_X.npy`, `val_y.npy`, `test_X.npy`, `test_y.npy`

#### Step 3: 训练模型

```bash
python scripts/03_train.py \
    --feature_dir ./data/features \
    --config configs/default.yaml \
    --save_dir ./checkpoints
```

默认使用简化版 Transformer 分类器，效果通常足够好且无需额外依赖。

#### Step 4: 推理检测

```bash
# 对单个视频
python scripts/04_inference.py \
    --video ./data/raw_videos/test.mp4 \
    --model ./checkpoints/best_model.pt \
    --save_dir ./outputs

# 批量推理
python scripts/04_inference.py \
    --video_dir ./data/raw_videos \
    --model ./checkpoints/best_model.pt \
    --save_dir ./outputs
```

---

## 核心设计

### 时序特征（12维）

对每帧提取描述交通群体状态的12维特征：

| 维度 | 特征名 | 说明 |
|------|--------|------|
| 0 | vehicle_count | 车辆数（归一化） |
| 1 | avg_speed | 平均速度 |
| 2 | speed_std | 速度标准差 |
| 3 | stop_ratio | 停止车辆比例 |
| 4-5 | avg_center_x/y | 车辆中心位置均值 |
| 6 | density | 车辆密度 |
| 7 | direction_entropy | 运动方向混乱度（熵） |
| 8-9 | x/y_velocity_mean | 平均速度向量 |
| 10 | interaction_score | 车辆交互强度 |
| 11 | tracklet_churn | 新旧车辆更替率 |

通过滑动窗口（默认30帧）构造 `[seq_len, feat_dim]` 的时序样本，输入分类模型。

### 模型选择

配置文件 `configs/default.yaml` 中 `model.type` 控制：

- **`simple_tsc`** (默认): 简化版 Transformer Encoder，轻量、无需预训练、即开即用
- **`units`**: 官方 UniTS 模型（需要克隆官方代码并放置到 `src/models/units_official/`）

切换方式：

```yaml
model:
  type: "units"   # 或 "simple_tsc"
```

#### 使用官方 UniTS（可选）

```bash
cd /tmp
git clone https://github.com/mims-harvard/UniTS.git
cp -r UniTS/models src/models/units_official
pip install timm einops
```

然后修改配置 `model.type: "units"` 即可。

---

## 自定义配置

`configs/default.yaml` 关键参数：

```yaml
tracking:
  yolo_model: "yolov8n.pt"     # 检测模型，可选 n/s/m/l
  conf_thresh: 0.3             # 检测置信度
  vehicle_classes: [2,3,5,7]   # 只跟踪 car/motorcycle/bus/truck

features:
  window_size: 30              # 时间序列长度（秒/帧）
  feat_dim: 12                 # 特征维度

events:
  classes:                     # 自定义事件类别
    - "normal"
    - "congestion"
    - "accident"
    ...

model:
  type: "simple_tsc"           # 或 "units"
  simple_tsc:
    d_model: 128
    num_layers: 2

training:
  epochs: 50
  batch_size: 32
  lr: 0.001
```

---

## 注意事项

1. **关于标签**：本项目需要你对视频片段进行事件标注。如果没有现成标注：
   - 先用 `00_prepare_annotations.py` 生成模板
   - 根据视频内容人工标注类别索引
   - 标注量取决于事件类型的区分难度，建议每类至少30-50个样本

2. **关于无监督/半监督**：如果你有大量无标签视频，可以：
   - 先用简化版模型在少量标注上训练一个基线
   - 用基线模型对未标注数据打伪标签
   - 筛选高置信度伪标签加入训练集迭代优化

3. **关于性能**：
   - 视频跟踪是计算瓶颈，建议使用GPU
   - 模型推理很快，单条时序样本在CPU上<10ms

---

## 扩展思路

- **更精细的特征**：加入车道线信息、车辆间距方差、急刹车检测等
- **多摄像头融合**：同一道路多个摄像头的时序拼接
- **在线检测**：对实时视频流逐帧输出事件概率
- **异常检测**：用自编码器或对比学习做无监督异常发现，减少标注依赖
