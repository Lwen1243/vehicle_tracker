"""
通用工具函数
"""
import os
import json
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """设置随机种子保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: str):
    """保存JSON文件"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """加载JSON文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_video_files(directory: str, extensions: tuple = (".mp4", ".avi", ".mkv", ".mov")) -> List[str]:
    """获取目录下所有视频文件路径"""
    video_files = []
    for ext in extensions:
        video_files.extend(Path(directory).glob(f"*{ext}"))
    return sorted([str(p) for p in video_files])


def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    计算两个bbox的IoU
    box format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    
    if boxAArea + boxBArea - interArea <= 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)
