"""
UniTS 官方模型适配器

官方 UniTS 代码已内置在 src/models/units_official/ 中
输入格式与官方一致: [batch_size, seq_len, enc_in]
"""
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# 将官方代码目录加入路径
_UNITS_PATH = os.path.join(os.path.dirname(__file__), "units_official")
if _UNITS_PATH not in sys.path:
    sys.path.insert(0, _UNITS_PATH)

from UniTS import Model as UniTSModel


class UniTSArgs:
    """UniTS 模型需要的参数对象"""
    def __init__(self, config: Dict[str, Any]):
        model_cfg = config.get("model", {}).get("units", {})
        self.d_model = model_cfg.get("d_model", 128)
        self.n_heads = model_cfg.get("n_heads", 8)
        self.e_layers = model_cfg.get("e_layers", 2)
        self.patch_len = model_cfg.get("patch_len", 8)
        self.stride = model_cfg.get("stride", 8)
        self.prompt_num = model_cfg.get("prompt_num", 3)
        self.dropout = model_cfg.get("dropout", 0.1)
        self.share_embedding = False
        self.large_model = False


class UniTSAdapter(nn.Module):
    """
    适配本项目特征格式到 UniTS 官方模型的包装器
    
    官方 UniTS 分类输入:
        x_enc: [batch_size, seq_len, enc_in]
        x_mark_enc: 时间戳标记（dummy）
        task_id: int
        task_name: "classification"
    
    输出:
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
        
        # 构建 args
        args = UniTSArgs({"model": {"units": units_config}} if units_config else {})
        
        # 构建 configs_list
        config = {
            "task_name": "classification",
            "dataset": "TrafficEvent",
            "data": "UEA",
            "embed": "timeF",
            "root_path": "./data/features",
            "seq_len": seq_len,
            "label_len": 0,
            "pred_len": 0,
            "enc_in": feat_dim,
            "num_class": num_classes,
            "c_out": None,
        }
        configs_list = [["CLS_TrafficEvent", config]]
        
        # 初始化官方 UniTS 模型
        self.model = UniTSModel(args=args, configs_list=configs_list, pretrain=False)
        
        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[UniTSAdapter] 加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            self.model.load_state_dict(state_dict, strict=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, feat_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # dummy time features [B, T, 4]
        x_mark = torch.zeros(batch_size, self.seq_len, 4, device=device)
        
        out = self.model(
            x_enc=x,
            x_mark_enc=x_mark,
            task_id=0,
            task_name="classification"
        )
        return out
