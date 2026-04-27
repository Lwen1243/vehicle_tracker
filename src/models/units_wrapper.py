"""
UniTS 官方模型适配器

说明:
- 本文件提供一个适配层，使官方 UniTS 模型可以接收本项目生成的时序特征
- 使用前需要从 https://github.com/mims-harvard/UniTS 下载官方代码并放到项目目录
- 或者将官方 models/UniTS.py 及相关依赖复制到 src/models/units_official/ 下

如果不想使用官方 UniTS，可以直接用 simple_tsc.py 中的简化版 Transformer，
效果通常足够好，且无需额外依赖。
"""
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class UniTSAdapter(nn.Module):
    """
    适配本项目特征格式到 UniTS 官方模型的包装器
    
    官方 UniTS 输入:
        x_enc: [batch_size, seq_len, enc_in]  (时间序列样本)
        x_mark_enc: 时间戳标记（分类任务可不传）
        task_id: int, 任务索引
        task_name: str = "classification"
    
    官方 UniTS 分类输出:
        [batch_size, num_classes]
    """
    
    def __init__(
        self,
        feat_dim: int,
        seq_len: int,
        num_classes: int,
        units_config: Optional[Dict[str, Any]] = None,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # 尝试导入官方 UniTS
        self.units_model = self._build_units_model(units_config, pretrained_path)
    
    def _build_units_model(self, units_config, pretrained_path):
        """构建或加载官方 UniTS 模型"""
        try:
            # 尝试从本地加载官方代码
            units_path = os.path.join(os.path.dirname(__file__), "units_official")
            if os.path.exists(units_path) and units_path not in sys.path:
                sys.path.insert(0, units_path)
            
            from models.UniTS import Model as UniTSModel
        except ImportError:
            print("[UniTSAdapter] 警告: 未找到官方 UniTS 代码，将使用简化版模型替代")
            print("  如需使用官方 UniTS，请执行:")
            print("  git clone https://github.com/mims-harvard/UniTS.git")
            print("  cp -r UniTS/models src/models/units_official")
            print("  并安装依赖: timm, einops")
            
            # fallback 到简化版
            from .simple_tsc import SimpleTSClassifier
            return SimpleTSClassifier(
                feat_dim=self.feat_dim,
                seq_len=self.seq_len,
                num_classes=self.num_classes,
            )
        
        # 构造 UniTS 需要的 configs_list 格式
        # configs_list 是 [[task_data_name, config_dict], ...]
        config = units_config or {
            "task_name": "classification",
            "dataset": "TrafficEvent",
            "data": "UEA",
            "embed": "timeF",
            "root_path": "./data/features",
            "seq_len": self.seq_len,
            "label_len": 0,
            "pred_len": 0,
            "enc_in": self.feat_dim,
            "num_class": self.num_classes,
            "c_out": None,
        }
        configs_list = [["CLS_TrafficEvent", config]]
        
        # 构造 args 对象
        class Args:
            d_model = units_config.get("d_model", 128) if units_config else 128
            patch_len = units_config.get("patch_len", 4) if units_config else 4
            stride = units_config.get("stride", 2) if units_config else 2
            prompt_num = units_config.get("prompt_num", 3) if units_config else 3
            e_layers = units_config.get("e_layers", 2) if units_config else 2
            n_heads = units_config.get("n_heads", 8) if units_config else 8
            dropout = units_config.get("dropout", 0.1) if units_config else 0.1
            share_embedding = False
            large_model = False
        
        args = Args()
        model = UniTSModel(configs_list=configs_list, args=args)
        
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[UniTSAdapter] 加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, feat_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # 检查底层模型类型
        if hasattr(self.units_model, 'forward') and 'task_name' in self.units_model.forward.__code__.co_varnames:
            # 官方 UniTS 模型
            x_mark = torch.zeros(batch_size, self.seq_len, 4, device=x.device)  # dummy time features
            out = self.units_model(
                x_enc=x,
                x_mark_enc=x_mark,
                task_id=0,
                task_name="classification"
            )
            return out
        else:
            # 简化版模型
            return self.units_model(x)


def check_units_available() -> bool:
    """检查官方 UniTS 是否可用"""
    try:
        units_path = os.path.join(os.path.dirname(__file__), "units_official")
        if os.path.exists(units_path) and units_path not in sys.path:
            sys.path.insert(0, units_path)
        from models.UniTS import Model
        return True
    except ImportError:
        return False
