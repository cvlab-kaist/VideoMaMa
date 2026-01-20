#!/usr/bin/env python3
"""
Script to delete unused video files from the videomama folder
based on what's referenced in index.html
"""

import os
from pathlib import Path

# Videos referenced in bentoData (from index.html)
referenced_videos = {
    # ID 1 - cheerleader
    "0_cheerleader_rgb.mp4",
    "0_cheerleader_mask.mp4",
    "0_cheerleader_pred_alpha.mp4",
    "0_cheerleader_pred_greenscreen.mp4",
    "0_cheerleader_mask_greenscreen.mp4",  # Also include mask_greenscreen variants
    
    # ID 2 - ad1aa270
    "0_ad1aa270_rgb.mp4",
    "0_ad1aa270_mask.mp4",
    "0_ad1aa270_pred_alpha.mp4",
    "0_ad1aa270_pred_greenscreen.mp4",
    "0_ad1aa270_mask_greenscreen.mp4",
    
    # ID 3 - 8cfc2be8 (note: topLeft is missing in HTML, but we keep the set)
    "0_8cfc2be8_rgb.mp4",
    "0_8cfc2be8_mask.mp4",
    "0_8cfc2be8_pred_alpha.mp4",
    "0_8cfc2be8_pred_greenscreen.mp4",
    "0_8cfc2be8_mask_greenscreen.mp4",
    
    # ID 4 - 019020
    "019020_rgb.mp4",
    "019020_mask.mp4",
    "019020_pred_alpha.mp4",
    "019020_pred_greenscreen.mp4",
    "019020_mask_greenscreen.mp4",
    
    # ID 5 - 019031
    "019031_rgb.mp4",
    "019031_mask.mp4",
    "019031_pred_alpha.mp4",
    "019031_pred_greenscreen.mp4",
    "019031_mask_greenscreen.mp4",
    
    # ID 16 - 87ac746d
    "87ac746d_rgb.mp4",
    "87ac746d_mask.mp4",
    "87ac746d_pred_alpha.mp4",
    "87ac746d_pred_greenscreen.mp4",
    "87ac746d_mask_greenscreen.mp4",
    
    # ID 23 - videoplayback_3
    "videoplayback_3_rgb.mp4",
    "videoplayback_3_mask.mp4",
    "videoplayback_3_pred_alpha.mp4",
    "videoplayback_3_pred_greenscreen.mp4",
    "videoplayback_3_mask_greenscreen.mp4",
    
    # ID 24 - 019006
    "019006_rgb.mp4",
    "019006_mask.mp4",
    "019006_pred_alpha.mp4",
    "019006_pred_greenscreen.mp4",
    "019006_mask_greenscreen.mp4",
}

def main():
    videomama_dir = Path("videomama")
    
    if not videomama_dir.exists():
        print(f"Error: {videomama_dir} does not exist")
        return
    
    # Get all mp4 files in the directory
    all_videos = list(videomama_dir.glob("*.mp4"))
    
    print(f"Total videos in folder: {len(all_videos)}")
    print(f"Referenced videos: {len(referenced_videos)}")
    print()
    
    # Find unused videos
    unused_videos = []
    for video_path in all_videos:
        if video_path.name not in referenced_videos:
            unused_videos.append(video_path)
    
    print(f"Found {len(unused_videos)} unused videos:\n")
    
    if not unused_videos:
        print("No unused videos to delete!")
        return
    
    # Show unused videos
    for video in sorted(unused_videos):
        print(f"  {video.name}")
    
    print()
    response = input(f"Delete these {len(unused_videos)} unused videos? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        deleted_count = 0
        failed_count = 0
        
        for video in unused_videos:
            try:
                video.unlink()
                print(f"✓ Deleted: {video.name}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ Failed to delete {video.name}: {e}")
                failed_count += 1
        
        print()
        print(f"Summary: {deleted_count} deleted, {failed_count} failed")
        
        # Show remaining files
        remaining = list(videomama_dir.glob("*.mp4"))
        print(f"\nRemaining videos: {len(remaining)}")
    else:
        print("Deletion cancelled.")

if __name__ == "__main__":
    main()
