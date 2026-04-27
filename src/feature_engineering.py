"""
时序特征工程模块
将车辆跟踪轨迹转换为多变量时间序列，供 UniTS 使用。

核心思路:
- 对每一帧/时间窗口，提取描述交通状态的群体特征
- 使用滑动窗口构造固定长度的时间序列样本
- 输出格式: numpy array [num_samples, seq_len, feat_dim]

定义的特征维度（默认12维）:
  0.  vehicle_count       - 当前帧车辆数（归一化到max_tracklets）
  1.  avg_speed          - 车辆平均速度（像素/秒）
  2.  speed_std          - 速度标准差
  3.  stop_ratio         - 停止车辆比例
  4.  avg_center_x       - 车辆中心点x均值（归一化到图像宽度）
  5.  avg_center_y       - 车辆中心点y均值（归一化到图像高度）
  6.  density            - 车辆密度（单位面积车辆数，基于凸包或网格）
  7.  direction_entropy  - 运动方向熵（混乱度）
  8.  x_velocity_mean    - x方向平均速度
  9.  y_velocity_mean    - y方向平均速度
  10. interaction_score  - 车辆交互强度（平均最近邻距离的倒数）
  11. tracklet_churn     - 新旧tracklet更替率
"""
import os
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
from scipy.spatial.distance import cdist
from scipy.stats import entropy


class TrajectoryFeatureExtractor:
    """从轨迹数据提取时序特征"""
    
    def __init__(self, config: Dict[str, Any], image_size: Tuple[int, int] = (1920, 1080)):
        self.cfg = config["features"]
        self.window_size = self.cfg["window_size"]
        self.stride = self.cfg["stride"]
        self.feat_dim = self.cfg["feat_dim"]
        self.max_tracklets = self.cfg["max_tracklets"]
        self.stop_speed_thresh = self.cfg["stop_speed_thresh"]
        self.img_w, self.img_h = image_size
        
        # 存储历史轨迹用于计算速度
        self.track_history = defaultdict(lambda: deque(maxlen=5))
        self.prev_frame_data = None
        self.prev_frame_id = -1
        
    def reset(self):
        """重置状态，用于处理新视频"""
        self.track_history.clear()
        self.prev_frame_data = None
        self.prev_frame_id = -1
    
    def extract_frame_features(self, frame_data: Dict[str, Any]) -> np.ndarray:
        """
        从单帧跟踪数据提取特征向量 [feat_dim]
        """
        detections = frame_data.get("detections", [])
        n = len(detections)
        
        if n == 0:
            return np.zeros(self.feat_dim, dtype=np.float32)
        
        # 收集当前帧数据
        centers = np.array([d["center"] for d in detections])  # [n, 2]
        bboxes = np.array([d["bbox"] for d in detections])      # [n, 4]
        track_ids = [d["track_id"] for d in detections]
        
        # 计算速度
        speeds = []
        directions = []
        velocities = []
        
        for i, tid in enumerate(track_ids):
            self.track_history[tid].append({
                "frame_id": frame_data["frame_id"],
                "center": centers[i],
                "timestamp": frame_data.get("timestamp", frame_data["frame_id"])
            })
            
            hist = self.track_history[tid]
            if len(hist) >= 2:
                # 用最近两帧计算瞬时速度
                dt = hist[-1]["timestamp"] - hist[-2]["timestamp"]
                if dt > 0:
                    dx = hist[-1]["center"][0] - hist[-2]["center"][0]
                    dy = hist[-1]["center"][1] - hist[-2]["center"][1]
                    v = math.sqrt(dx**2 + dy**2) / dt
                    speeds.append(v)
                    velocities.append([dx / dt, dy / dt])
                    if v > 0.1:
                        directions.append(math.atan2(dy, dx))
        
        # 初始化特征
        feats = np.zeros(self.feat_dim, dtype=np.float32)
        
        # 0. 车辆数（归一化）
        feats[0] = min(n / self.max_tracklets, 1.0)
        
        # 1-3. 速度统计
        if speeds:
            feats[1] = min(np.mean(speeds) / 50.0, 1.0)  # 假设最大速度50px/s
            feats[2] = min(np.std(speeds) / 30.0, 1.0)
            feats[3] = np.mean([1.0 if s < self.stop_speed_thresh else 0.0 for s in speeds])
        
        # 4-5. 中心位置（归一化）
        feats[4] = np.mean(centers[:, 0]) / self.img_w
        feats[5] = np.mean(centers[:, 1]) / self.img_h
        
        # 6. 密度：使用bbox覆盖面积占比近似
        if n > 1:
            # 简单密度：所有bbox并集面积的倒数 * n
            total_bbox_area = 0
            for bbox in bboxes:
                area = max(0, bbox[2]-bbox[0]) * max(0, bbox[3]-bbox[1])
                total_bbox_area += area
            img_area = self.img_w * self.img_h
            density = n / (total_bbox_area / img_area + 1e-6)
            feats[6] = min(density / 10.0, 1.0)
        else:
            feats[6] = 0.0
        
        # 7. 方向熵
        if len(directions) >= 2:
            # 将方向分成8个bin计算熵
            bins = np.histogram(directions, bins=8, range=(-np.pi, np.pi))[0]
            bins = bins / (bins.sum() + 1e-10)
            feats[7] = entropy(bins) / np.log(8)  # 归一化到[0,1]
        else:
            feats[7] = 0.0
        
        # 8-9. 平均速度向量
        if velocities:
            vel_arr = np.array(velocities)
            feats[8] = np.mean(vel_arr[:, 0]) / 50.0  # 归一化
            feats[9] = np.mean(vel_arr[:, 1]) / 50.0
        
        # 10. 交互强度：平均最近邻距离的归一化倒数
        if n > 1:
            dists = cdist(centers, centers, 'euclidean')
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            avg_min_dist = np.mean(min_dists)
            # 距离越近，交互越强
            feats[10] = min(100.0 / (avg_min_dist + 10.0), 1.0)
        else:
            feats[10] = 0.0
        
        # 11. tracklet更替率（当前帧新出现或消失的tracklet比例）
        if self.prev_frame_data is not None:
            prev_ids = set(d["track_id"] for d in self.prev_frame_data.get("detections", []))
            curr_ids = set(track_ids)
            churn = len(prev_ids.symmetric_difference(curr_ids)) / max(len(prev_ids), len(curr_ids), 1)
            feats[11] = churn
        else:
            feats[11] = 0.0
        
        self.prev_frame_data = frame_data
        return feats
    
    def process_trajectory_file(
        self,
        traj_path: str,
        label: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        处理单个轨迹文件，生成滑动窗口时序样本
        
        Args:
            traj_path: 轨迹json文件路径
            label: 该视频的事件标签（None表示无标签）
            
        Returns:
            X: [num_samples, seq_len, feat_dim] 时序特征
            y: [num_samples] 标签（如果label不为None）
        """
        self.reset()
        
        with open(traj_path, "r", encoding="utf-8") as f:
            frames = json.load(f)
        
        if len(frames) < self.window_size:
            print(f"[FeatureExtractor] 警告: {traj_path} 帧数({len(frames)})少于窗口大小({self.window_size})，跳过")
            return np.array([]), np.array([]) if label is not None else None
        
        # 逐帧提取特征
        frame_feats = []
        for frame in frames:
            feat = self.extract_frame_features(frame)
            frame_feats.append(feat)
        
        frame_feats = np.array(frame_feats, dtype=np.float32)  # [T, feat_dim]
        
        # 滑动窗口
        num_samples = (len(frame_feats) - self.window_size) // self.stride + 1
        X = np.zeros((num_samples, self.window_size, self.feat_dim), dtype=np.float32)
        
        for i in range(num_samples):
            start = i * self.stride
            end = start + self.window_size
            X[i] = frame_feats[start:end]
        
        if label is not None:
            y = np.full(num_samples, label, dtype=np.int64)
            return X, y
        
        return X, None
    
    def build_dataset(
        self,
        traj_dir: str,
        annotations: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        批量构建数据集
        
        Args:
            traj_dir: 轨迹文件目录
            annotations: {文件名(不含扩展名): 标签}，如 {"video_001": 1}
            
        Returns:
            X: [total_samples, seq_len, feat_dim]
            y: [total_samples] 或 None
        """
        traj_files = sorted(Path(traj_dir).glob("*.json"))
        if not traj_files:
            raise ValueError(f"{traj_dir} 中没有找到轨迹文件")
        
        all_X = []
        all_y = []
        
        for traj_path in traj_files:
            stem = traj_path.stem
            label = annotations.get(stem, None) if annotations else None
            
            X, y = self.process_trajectory_file(str(traj_path), label=label)
            if X.size > 0:
                all_X.append(X)
                if y is not None:
                    all_y.append(y)
        
        if not all_X:
            raise ValueError("没有生成任何有效样本，请检查轨迹数据和窗口大小")
        
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0) if all_y else None
        
        print(f"[FeatureExtractor] 数据集构建完成: X.shape={X.shape}, 样本数={X.shape[0]}")
        if y is not None:
            print(f"[FeatureExtractor] 标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y


def save_features(
    X: np.ndarray,
    y: Optional[np.ndarray],
    save_dir: str,
    prefix: str = "train"
):
    """保存特征和标签为npy文件"""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{prefix}_X.npy"), X)
    if y is not None:
        np.save(os.path.join(save_dir, f"{prefix}_y.npy"), y)
    print(f"[FeatureExtractor] 特征已保存到 {save_dir}: {prefix}_X.npy")


def load_features(save_dir: str, prefix: str = "train") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """加载特征和标签"""
    X = np.load(os.path.join(save_dir, f"{prefix}_X.npy"))
    y_path = os.path.join(save_dir, f"{prefix}_y.npy")
    y = np.load(y_path) if os.path.exists(y_path) else None
    return X, y
