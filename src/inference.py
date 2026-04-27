"""
推理脚本：对视频或时序特征进行事件分类
"""
import os
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .video_tracker import VideoTracker
from .feature_engineering import TrajectoryFeatureExtractor
from .models.simple_tsc import SimpleTSClassifier
from .utils import load_config


class EventDetector:
    """道路事件检测器（端到端）"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/default.yaml",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = load_config(config_path)
        self.class_names = self.config["events"]["classes"]
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 初始化跟踪器和特征提取器
        self.tracker = VideoTracker(self.config)
        self.extractor = TrajectoryFeatureExtractor(self.config)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        saved_config = checkpoint.get("config", self.config)
        num_classes = checkpoint.get("num_classes", len(self.class_names))
        
        # 确定模型类型
        model_type = saved_config.get("model", {}).get("type", "simple_tsc")
        feat_dim = saved_config["features"]["feat_dim"]
        seq_len = saved_config["features"]["window_size"]
        
        if model_type == "units":
            try:
                from .models.units_wrapper import UniTSAdapter
                model = UniTSAdapter(feat_dim, seq_len, num_classes)
            except Exception as e:
                print(f"[Inference] UniTS加载失败，回退到简化版: {e}")
                model = SimpleTSClassifier(feat_dim, seq_len, num_classes)
        else:
            model = SimpleTSClassifier(feat_dim, seq_len, num_classes)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        print(f"[Inference] 模型加载完成: {model_path}")
        print(f"[Inference] 类别: {self.class_names[:num_classes]}")
        return model
    
    def predict_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对时序特征进行预测
        
        Args:
            X: [num_samples, seq_len, feat_dim]
        Returns:
            pred_labels: [num_samples]
            pred_probs: [num_samples, num_classes]
        """
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        
        pred_labels = torch.argmax(probs, dim=1).cpu().numpy()
        pred_probs = probs.cpu().numpy()
        return pred_labels, pred_probs
    
    def predict_video(
        self,
        video_path: str,
        save_dir: Optional[str] = None,
        aggregation: str = "vote",
    ) -> Dict[str, Any]:
        """
        对单个视频进行端到端事件检测
        
        Args:
            video_path: 视频路径
            save_dir: 中间结果保存目录
            aggregation: 多窗口预测聚合方式，"vote"投票 或 "max_prob"最大概率
            
        Returns:
            result: {
                "video": video_path,
                "event": str,           # 预测的事件名称
                "confidence": float,    # 置信度
                "window_predictions": List[Dict],  # 每个窗口的预测
            }
        """
        print(f"[EventDetector] 处理视频: {video_path}")
        
        # 1. 跟踪
        traj_data = self.tracker.track_video(
            video_path,
            save_dir=save_dir,
        )
        
        if len(traj_data) < self.config["features"]["window_size"]:
            return {
                "video": video_path,
                "event": "unknown",
                "confidence": 0.0,
                "reason": "视频太短，无法提取完整时间窗口",
                "window_predictions": [],
            }
        
        # 2. 特征提取
        self.extractor.reset()
        frame_feats = []
        for frame in traj_data:
            feat = self.extractor.extract_frame_features(frame)
            frame_feats.append(feat)
        
        frame_feats = np.array(frame_feats, dtype=np.float32)
        
        # 构造滑动窗口
        window_size = self.config["features"]["window_size"]
        stride = self.config["features"]["stride"]
        num_samples = (len(frame_feats) - window_size) // stride + 1
        
        X = np.zeros((num_samples, window_size, frame_feats.shape[1]), dtype=np.float32)
        for i in range(num_samples):
            start = i * stride
            X[i] = frame_feats[start:start+window_size]
        
        # 3. 预测
        pred_labels, pred_probs = self.predict_features(X)
        
        # 4. 聚合结果
        window_results = []
        for i in range(num_samples):
            label_idx = pred_labels[i]
            window_results.append({
                "window_id": i,
                "start_frame": i * stride,
                "end_frame": i * stride + window_size,
                "event": self.class_names[label_idx] if label_idx < len(self.class_names) else "unknown",
                "confidence": float(pred_probs[i][label_idx]),
                "prob_distribution": {
                    self.class_names[j]: float(pred_probs[i][j])
                    for j in range(len(self.class_names))
                    if j < pred_probs.shape[1]
                }
            })
        
        if aggregation == "vote":
            # 多数投票
            unique, counts = np.unique(pred_labels, return_counts=True)
            final_label = unique[np.argmax(counts)]
            confidence = float(np.max(counts) / len(pred_labels))
        elif aggregation == "max_prob":
            # 取平均概率最高的类别
            mean_probs = np.mean(pred_probs, axis=0)
            final_label = int(np.argmax(mean_probs))
            confidence = float(mean_probs[final_label])
        else:
            final_label = int(np.bincount(pred_labels).argmax())
            confidence = 1.0
        
        result = {
            "video": video_path,
            "event": self.class_names[final_label] if final_label < len(self.class_names) else "unknown",
            "confidence": round(confidence, 4),
            "num_windows": num_samples,
            "window_predictions": window_results,
        }
        
        # 保存结果
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            result_path = os.path.join(save_dir, f"{Path(video_path).stem}_result.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[EventDetector] 结果已保存: {result_path}")
        
        print(f"[EventDetector] 检测结果: {result['event']} (置信度: {result['confidence']:.3f})")
        return result
    
    def predict_trajectory_file(self, traj_path: str) -> Dict[str, Any]:
        """
        对已保存的轨迹文件进行事件检测
        """
        self.extractor.reset()
        
        import json
        with open(traj_path, "r") as f:
            frames = json.load(f)
        
        frame_feats = []
        for frame in frames:
            feat = self.extractor.extract_frame_features(frame)
            frame_feats.append(feat)
        
        frame_feats = np.array(frame_feats, dtype=np.float32)
        
        window_size = self.config["features"]["window_size"]
        stride = self.config["features"]["stride"]
        
        if len(frame_feats) < window_size:
            return {"event": "unknown", "reason": "too_short"}
        
        num_samples = (len(frame_feats) - window_size) // stride + 1
        X = np.zeros((num_samples, window_size, frame_feats.shape[1]), dtype=np.float32)
        for i in range(num_samples):
            start = i * stride
            X[i] = frame_feats[start:start+window_size]
        
        pred_labels, pred_probs = self.predict_features(X)
        
        mean_probs = np.mean(pred_probs, axis=0)
        final_label = int(np.argmax(mean_probs))
        
        return {
            "event": self.class_names[final_label] if final_label < len(self.class_names) else "unknown",
            "confidence": float(mean_probs[final_label]),
            "prob_distribution": {
                self.class_names[j]: float(mean_probs[j])
                for j in range(min(len(self.class_names), len(mean_probs)))
            }
        }
