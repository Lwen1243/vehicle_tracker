#!/usr/bin/env python3
"""
第零步：生成标注模板文件（辅助脚本）

说明：
- 如果你已经有标注，直接使用 02_build_features.py 的 --annotations 参数
- 如果没有标注，先运行本脚本生成模板，然后人工填写类别索引

用法:
    python scripts/00_prepare_annotations.py \
        --traj_dir ./data/trajectories \
        --output ./data/annotations.json
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, ".")

from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="生成标注模板")
    parser.add_argument("--traj_dir", type=str, required=True, help="轨迹目录")
    parser.add_argument("--output", type=str, default="./data/annotations.json", help="输出标注文件")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    class_names = config["events"]["classes"]
    
    traj_files = sorted(Path(args.traj_dir).glob("*.json"))
    if not traj_files:
        print(f"[Error] {args.traj_dir} 中没有轨迹文件")
        return
    
    template = {}
    for traj_path in traj_files:
        template[traj_path.stem] = 0  # 默认全部标为0 (normal)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    print(f"[Done] 标注模板已生成: {args.output}")
    print(f"[Info] 共 {len(template)} 个样本")
    print(f"[Info] 事件类别定义:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print(f"\n请编辑 {args.output} 文件，将对应视频的类别索引修改为正确值后，")
    print("再运行 scripts/02_build_features.py --annotations 参数进行特征构建。")


if __name__ == "__main__":
    main()
