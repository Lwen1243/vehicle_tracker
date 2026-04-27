"""
训练脚本：时间序列分类模型训练
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from .dataset import TrafficEventDataset, Augmentation
from .models.simple_tsc import SimpleTSClassifier
from .utils import set_seed, ensure_dir


def get_model(config: Dict[str, Any], num_classes: int):
    """根据配置获取模型"""
    model_cfg = config["model"]
    feat_dim = config["features"]["feat_dim"]
    seq_len = config["features"]["window_size"]
    
    if model_cfg["type"] == "units":
        try:
            from .models.units_wrapper import UniTSAdapter
            return UniTSAdapter(
                feat_dim=feat_dim,
                seq_len=seq_len,
                num_classes=num_classes,
                units_config=model_cfg.get("units"),
            )
        except Exception as e:
            print(f"[Train] UniTS 加载失败，回退到简化版模型: {e}")
    
    # 默认使用简化版
    simple_cfg = model_cfg.get("simple_tsc", {})
    return SimpleTSClassifier(
        feat_dim=feat_dim,
        seq_len=seq_len,
        num_classes=num_classes,
        d_model=simple_cfg.get("d_model", 128),
        n_heads=simple_cfg.get("n_heads", 4),
        num_layers=simple_cfg.get("num_layers", 2),
        dim_feedforward=simple_cfg.get("dim_feedforward", 256),
        dropout=simple_cfg.get("dropout", 0.2),
    )


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, criterion, device: str) -> float:
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: str, num_classes: int):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    
    # 多分类指标
    if num_classes > 2:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


def train_model(
    config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_dir: str = "./checkpoints",
    class_names: Optional[list] = None,
) -> str:
    """
    训练分类模型
    
    Args:
        config: 配置字典
        X_train: 训练特征 [N, T, D]
        y_train: 训练标签 [N]
        X_val: 验证特征（可选）
        y_val: 验证标签（可选）
        save_dir: 模型保存目录
        class_names: 类别名称列表
        
    Returns:
        best_model_path: 最佳模型路径
    """
    set_seed(config["training"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] 使用设备: {device}")
    
    num_classes = len(np.unique(y_train))
    print(f"[Train] 类别数: {num_classes}, 训练样本: {len(y_train)}")
    
    # 数据增强
    aug_config = config["training"].get("augment", {})
    transform = Augmentation(
        noise_std=aug_config.get("noise_std", 0.01),
        scale_range=tuple(aug_config.get("scale_range", [0.95, 1.05]))
    ) if aug_config else None
    
    train_dataset = TrafficEventDataset(X_train, y_train, transform=transform)
    
    # 划分验证集
    val_dataset = None
    if X_val is not None and y_val is not None:
        val_dataset = TrafficEventDataset(X_val, y_val)
    else:
        val_ratio = config["training"].get("val_ratio", 0.2)
        if val_ratio > 0:
            val_size = int(len(train_dataset) * val_ratio)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
    
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    
    # 模型
    model = get_model(config, num_classes).to(device)
    print(f"[Train] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0001)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()
    
    # 早停
    early_stop = EarlyStopping(patience=config["training"].get("patience", 10))
    
    # 训练循环
    best_val_f1 = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pt")
    ensure_dir(save_dir)
    
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        log_str = f"Epoch {epoch:3d}/{config['training']['epochs']} | Train Loss: {train_loss:.4f}"
        
        if val_loader:
            val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
            log_str += f" | Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}"
            
            # 保存最佳模型
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'num_classes': num_classes,
                    'class_names': class_names,
                    'val_metrics': val_metrics,
                }, best_model_path)
            
            # 早停检查
            if early_stop(val_metrics['loss'], model):
                print(f"\n[Train] 早停触发于 epoch {epoch}")
                break
        
        log_str += f" | Time: {time.time()-start_time:.1f}s"
        print(log_str)
    
    # 如果没有验证集，保存最后epoch的模型
    if not val_loader:
        torch.save({
            'epoch': config["training"]["epochs"],
            'model_state_dict': model.state_dict(),
            'config': config,
            'num_classes': num_classes,
            'class_names': class_names,
        }, best_model_path)
    
    print(f"\n[Train] 训练完成，最佳模型保存于: {best_model_path}")
    if val_loader:
        print(f"[Train] 最佳验证 F1: {best_val_f1:.4f}")
    
    # 打印详细报告
    if val_loader and class_names:
        model.load_state_dict(early_stop.best_weights)
        final_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        print("\n[Train] 最终验证集分类报告:")
        print(classification_report(
            final_metrics['labels'], final_metrics['predictions'],
            target_names=class_names, zero_division=0
        ))
    
    return best_model_path
