"""
视频车辆检测与跟踪模块
基于 Ultralytics YOLOv8 + ByteTrack

输出格式（每帧一个字典）:
{
    "frame_id": int,
    "timestamp": float,  # 秒
    "detections": [
        {
            "track_id": int,
            "bbox": [x1, y1, x2, y2],
            "class_id": int,
            "class_name": str,
            "conf": float,
            "center": [cx, cy]
        }, ...
    ]
}
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm


class VideoTracker:
    """视频车辆跟踪器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config["tracking"]
        self.yolo_model = None
        self._load_model()
        
    def _load_model(self):
        """懒加载YOLO模型"""
        if self.yolo_model is None:
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError(
                    "请安装 ultralytics: pip install ultralytics"
                )
            model_path = self.cfg["yolo_model"]
            print(f"[VideoTracker] 正在加载模型: {model_path}")
            self.yolo_model = YOLO(model_path)
            print(f"[VideoTracker] 模型加载完成")
    
    def track_video(
        self,
        video_path: str,
        save_dir: Optional[str] = None,
        save_vis: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        对单个视频进行跟踪
        
        Args:
            video_path: 视频文件路径
            save_dir: 轨迹json保存目录
            save_vis: 是否保存可视化视频
            
        Returns:
            frames_data: 每帧的跟踪结果列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # 处理帧率控制
        target_fps = self.cfg.get("fps")
        if target_fps is not None and target_fps > 0:
            frame_interval = max(1, int(fps / target_fps))
        else:
            frame_interval = 1
        
        # 运行跟踪
        print(f"[VideoTracker] 开始跟踪: {video_path}")
        print(f"  总帧数: {total_frames}, FPS: {fps:.1f}, 处理间隔: {frame_interval}")
        
        results = self.yolo_model.track(
            source=video_path,
            tracker=self.cfg["tracker"],
            conf=self.cfg["conf_thresh"],
            classes=self.cfg["vehicle_classes"],
            imgsz=self.cfg["imgsz"],
            verbose=False,
            stream=True,
        )
        
        frames_data = []
        frame_idx = 0
        processed_idx = 0
        
        # 可视化准备
        vis_writer = None
        if save_vis:
            vis_path = os.path.join(save_dir or ".", f"{Path(video_path).stem}_vis.mp4")
            ensure_dir(os.path.dirname(vis_path))
        
        for r in tqdm(results, total=total_frames, desc="Tracking"):
            timestamp = frame_idx / fps
            
            if frame_idx % frame_interval == 0:
                detections = []
                if r.boxes is not None and r.boxes.id is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    track_ids = r.boxes.id.cpu().numpy().astype(int)
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    names = r.names
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        detections.append({
                            "track_id": int(track_ids[i]),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "class_id": int(cls_ids[i]),
                            "class_name": names.get(int(cls_ids[i]), "unknown"),
                            "conf": float(confs[i]),
                            "center": [float(cx), float(cy)]
                        })
                
                frame_data = {
                    "frame_id": processed_idx,
                    "video_frame_id": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "detections": detections
                }
                frames_data.append(frame_data)
                processed_idx += 1
            
            frame_idx += 1
        
        print(f"[VideoTracker] 跟踪完成，共处理 {processed_idx} 帧，检测到 {sum(len(f['detections']) for f in frames_data)} 个目标")
        
        # 保存轨迹
        if save_dir:
            ensure_dir(save_dir)
            traj_path = os.path.join(save_dir, f"{Path(video_path).stem}.json")
            with open(traj_path, "w", encoding="utf-8") as f:
                json.dump(frames_data, f, ensure_ascii=False, indent=2)
            print(f"[VideoTracker] 轨迹已保存: {traj_path}")
        
        return frames_data
    
    def batch_track(
        self,
        video_dir: str,
        save_dir: str,
        save_vis: bool = False,
    ) -> List[str]:
        """
        批量处理视频目录
        
        Returns:
            保存的轨迹文件路径列表
        """
        from .utils import get_video_files, ensure_dir
        video_files = get_video_files(video_dir)
        if not video_files:
            print(f"[VideoTracker] 警告: {video_dir} 中没有找到视频文件")
            return []
        
        ensure_dir(save_dir)
        saved_paths = []
        
        for video_path in video_files:
            try:
                self.track_video(video_path, save_dir=save_dir, save_vis=save_vis)
                saved_paths.append(os.path.join(save_dir, f"{Path(video_path).stem}.json"))
            except Exception as e:
                print(f"[VideoTracker] 处理失败 {video_path}: {e}")
        
        return saved_paths


def ensure_dir(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
