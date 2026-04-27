"""
简化版时间序列分类器（基于 Transformer Encoder）

作为 UniTS 的轻量级替代方案，在没有 UniTS 预训练权重时可直接使用。
输入格式与 UniTS 分类任务兼容: [batch_size, seq_len, feat_dim]
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]


class SimpleTSClassifier(nn.Module):
    """
    基于 Transformer Encoder 的时间序列分类器
    
    Args:
        feat_dim: 输入特征维度
        seq_len: 序列长度
        num_classes: 类别数
        d_model: 模型维度
        n_heads: 注意力头数
        num_layers: Transformer层数
        dim_feedforward: FFN中间维度
        dropout: dropout率
    """
    
    def __init__(
        self,
        feat_dim: int,
        seq_len: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(feat_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, feat_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        # 输入投影
        x = self.input_proj(x)  # [B, T, D]
        
        # 加位置编码
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)  # [B, T, D]
        
        # 全局平均池化 + 最后一个token
        x_mean = x.mean(dim=1)      # [B, D]
        x_last = x[:, -1, :]        # [B, D]
        x = x_mean + x_last         # [B, D]
        
        # 分类
        x = self.norm(x)
        logits = self.classifier(x)  # [B, num_classes]
        
        return logits
