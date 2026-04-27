#!/usr/bin/env python3
"""
第四步：对新视频进行事件检测推理

用法:
    # 对单个视频推理
    python scripts/04_inference.py \
        --video ./data/raw_videos/test.mp4 \
        --model ./checkpoints/best_model.pt \
        --config configs/default.yaml \
        --save_dir ./outputs
    
    # 批量推理整个目录
    python scripts/04_inference.py \
        --video_dir ./data/raw_videos \
        --model ./checkpoints/best_model.pt \
        --config configs/default.yaml \
        --save_dir ./outputs
    
    # 对轨迹文件推理（跳过跟踪）
    python scripts/04_inference.py \
        --traj ./data/trajectories/test.json \
        --model ./checkpoints/best_model.pt \
        --config configs/default.yaml
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, ".")

from src.utils import load_config, get_video_files
from src.inference import EventDetector


def main():
    parser = argparse.ArgumentParser(description="事件检测推理")
    parser.add_argument("--video", type=str, default=None, help="单个视频路径")
    parser.add_argument("--video_dir", type=str, default=None, help="视频目录（批量推理）")
    parser.add_argument("--traj", type=str, default=None, help="单个轨迹文件路径")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir", type=str, default="./outputs", help="结果保存目录")
    parser.add_argument("--agg", type=str, default="vote", choices=["vote", "max_prob"],
                        help="多窗口聚合方式")
    args = parser.parse_args()
    
    detector = EventDetector(args.model, args.config)
    
    results = []
    
    # 批量视频推理
    if args.video_dir:
        video_files = get_video_files(args.video_dir)
        print(f"[Inference] 找到 {len(video_files)} 个视频文件")
        for vpath in video_files:
            result = detector.predict_video(vpath, save_dir=args.save_dir, aggregation=args.agg)
            results.append(result)
    
    # 单个视频推理
    elif args.video:
        result = detector.predict_video(args.video, save_dir=args.save_dir, aggregation=args.agg)
        results.append(result)
    
    # 轨迹文件推理
    elif args.traj:
        result = detector.predict_trajectory_file(args.traj)
        print(f"[Inference] 轨迹文件: {args.traj}")
        print(f"[Inference] 检测结果: {result['event']} (置信度: {result.get('confidence', 0):.3f})")
        results.append(result)
    
    else:
        print("[Error] 请提供 --video, --video_dir 或 --traj 之一")
        return
    
    # 保存汇总结果
    if results and args.save_dir:
        summary_path = Path(args.save_dir) / "inference_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[Inference] 汇总结果已保存: {summary_path}")
    
    # 打印统计
    if len(results) > 1:
        from collections import Counter
        events = [r["event"] for r in results if "event" in r]
        print("\n[Inference] 检测结果统计:")
        for evt, cnt in Counter(events).most_common():
            print(f"  {evt}: {cnt}")


if __name__ == "__main__":
    main()
