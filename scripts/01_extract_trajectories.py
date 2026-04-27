#!/usr/bin/env python3
"""
第一步：从监控视频提取车辆轨迹

用法:
    python scripts/01_extract_trajectories.py \
        --video_dir ./data/raw_videos \
        --save_dir ./data/trajectories \
        --config configs/default.yaml
"""
import sys
import argparse
sys.path.insert(0, ".")

from src.utils import load_config
from src.video_tracker import VideoTracker


def main():
    parser = argparse.ArgumentParser(description="从视频提取车辆轨迹")
    parser.add_argument("--video_dir", type=str, required=True, help="原始视频目录")
    parser.add_argument("--save_dir", type=str, default="./data/trajectories", help="轨迹保存目录")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    config = load_config(args.config)
    tracker = VideoTracker(config)
    
    tracker.batch_track(
        video_dir=args.video_dir,
        save_dir=args.save_dir,
        save_vis=False,
    )
    
    print("\n[Done] 轨迹提取完成！下一步: 02_build_features.py")


if __name__ == "__main__":
    main()
