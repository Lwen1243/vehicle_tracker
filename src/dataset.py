"""
PyTorch Dataset 定义
兼容 UniTS 和简化版分类器的输入格式
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable


class TrafficEventDataset(Dataset):
    """
    交通事件时序分类数据集
    
    数据格式:
        X: [num_samples, seq_len, feat_dim] 或 [num_samples, feat_dim, seq_len]
        y: [num_samples] 分类标签
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        channel_first: bool = False,  # UniTS默认不是channel_first，但有些模型需要
    ):
        """
        Args:
            X: 时序特征，shape [N, T, D] 或 [N, D, T]
            y: 标签，shape [N]
            transform: 数据增强函数
            channel_first: 是否将输出转为 [N, D, T]
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y is not None else None
        self.transform = transform
        self.channel_first = channel_first
        
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        
        if self.transform:
            x = self.transform(x)
        
        if self.channel_first and x.ndim == 2:
            # [T, D] -> [D, T]
            x = x.transpose(0, 1)
        
        if self.y is not None:
            return x, self.y[idx]
        return x, torch.tensor(-1)  # 无标签时返回-1


class Augmentation:
    """时序数据增强"""
    
    def __init__(self, noise_std: float = 0.01, scale_range: Tuple[float, float] = (0.95, 1.05)):
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 加噪
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        
        # 缩放
        if self.scale_range is not None:
            scale = np.random.uniform(*self.scale_range)
            x = x * scale
        
        return x
