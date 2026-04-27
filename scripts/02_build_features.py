#!/usr/bin/env python3
"""
第二步：从轨迹构建时序特征数据集

用法:
    # 无标签模式（用于生成未标注特征，后续可人工标注）
    python scripts/02_build_features.py \
        --traj_dir ./data/trajectories \
        --save_dir ./data/features \
        --config configs/default.yaml
    
    # 有标签模式（需提供标注文件）
    python scripts/02_build_features.py \
        --traj_dir ./data/trajectories \
        --save_dir ./data/features \
        --annotations ./data/annotations.json \
        --config configs/default.yaml

标注文件格式 (annotations.json):
{
    "video_001": 1,   // 视频名(不含扩展名) -> 事件类别索引
    "video_002": 0,
    ...
}
"""
import sys
import argparse
import json
sys.path.insert(0, ".")

from src.utils import load_config
from src.feature_engineering import TrajectoryFeatureExtractor, save_features


def main():
    parser = argparse.ArgumentParser(description="从轨迹构建时序特征")
    parser.add_argument("--traj_dir", type=str, required=True, help="轨迹文件目录")
    parser.add_argument("--save_dir", type=str, default="./data/features", help="特征保存目录")
    parser.add_argument("--annotations", type=str, default=None, help="标注文件路径(json)")
    parser.add_argument("--split", action="store_true", help="是否划分训练/验证/测试集")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    extractor = TrajectoryFeatureExtractor(config)
    
    # 加载标注
    annotations = None
    if args.annotations:
        with open(args.annotations, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        print(f"[BuildFeatures] 加载标注文件: {args.annotations}, 共 {len(annotations)} 条")
    
    # 构建数据集
    X, y = extractor.build_dataset(args.traj_dir, annotations)
    
    # 保存
    if args.split and y is not None:
        from sklearn.model_selection import train_test_split
        # 先划分出测试集
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=config["training"].get("test_ratio", 0.1),
            random_state=42, stratify=y
        )
        # 再划分训练/验证
        val_ratio = config["training"].get("val_ratio", 0.2)
        val_size = val_ratio / (1 - config["training"].get("test_ratio", 0.1))
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size,
            random_state=42, stratify=y_trainval
        )
        
        save_features(X_train, y_train, args.save_dir, "train")
        save_features(X_val, y_val, args.save_dir, "val")
        save_features(X_test, y_test, args.save_dir, "test")
        print(f"[BuildFeatures] 数据集划分完成:")
        print(f"  训练集: {len(y_train)} 样本")
        print(f"  验证集: {len(y_val)} 样本")
        print(f"  测试集: {len(y_test)} 样本")
    else:
        prefix = "unlabeled" if y is None else "all"
        save_features(X, y, args.save_dir, prefix)
    
    print("\n[Done] 特征构建完成！下一步:")
    if y is not None:
        print("  训练模型 -> scripts/03_train.py")
    else:
        print("  请先完成标注，再运行 scripts/03_train.py")


if __name__ == "__main__":
    main()
