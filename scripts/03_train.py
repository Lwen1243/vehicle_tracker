#!/usr/bin/env python3
"""
第三步：训练事件分类模型

用法:
    python scripts/03_train.py \
        --feature_dir ./data/features \
        --config configs/default.yaml \
        --save_dir ./checkpoints
"""
import sys
import argparse
sys.path.insert(0, ".")

from src.utils import load_config
from src.feature_engineering import load_features
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description="训练事件分类模型")
    parser.add_argument("--feature_dir", type=str, default="./data/features", help="特征目录")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="模型保存目录")
    args = parser.parse_args()
    
    config = load_config(args.config)
    class_names = config["events"]["classes"]
    
    # 加载数据
    print("[Train] 加载训练数据...")
    X_train, y_train = load_features(args.feature_dir, "train")
    
    # 尝试加载验证集
    try:
        X_val, y_val = load_features(args.feature_dir, "val")
    except FileNotFoundError:
        X_val, y_val = None, None
        print("[Train] 未找到验证集，将从训练集自动划分")
    
    # 训练
    best_path = train_model(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        save_dir=args.save_dir,
        class_names=class_names,
    )
    
    print(f"\n[Done] 训练完成！最佳模型: {best_path}")
    print("  下一步推理 -> scripts/04_inference.py")


if __name__ == "__main__":
    main()
