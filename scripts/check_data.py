# -*- coding: utf-8 -*-
"""
check_data.py - Verify lab_data folder structure for 5 tracking videos.

Usage:
    python scripts/check_data.py --lab-data-root C:/path/to/lab_data

    # Or use environment variable:
    $env:LAB_DATA = "C:/path/to/lab_data"
    python scripts/check_data.py --lab-data-root $env:LAB_DATA
"""

import argparse
import os
import sys
from pathlib import Path


def check_video(video_dir: Path, video_name: str) -> dict:
    """Check a single video directory and return status dict."""
    result = {
        "name": video_name,
        "exists": False,
        "img1_exists": False,
        "img_count": 0,
        "has_gt": False,
        "issues": [],
    }

    if not video_dir.exists():
        result["issues"].append(f"Directory not found: {video_dir}")
        return result

    result["exists"] = True

    # Check img1/
    img1_dir = video_dir / "img1"
    if not img1_dir.exists():
        result["issues"].append("Missing img1/ subdirectory")
    else:
        result["img1_exists"] = True
        jpg_files = (
            list(img1_dir.glob("*.jpg"))
            + list(img1_dir.glob("*.jpeg"))
            + list(img1_dir.glob("*.png"))
        )
        result["img_count"] = len(jpg_files)
        if result["img_count"] == 0:
            result["issues"].append("img1/ contains no image files (.jpg / .png)")

    # Check GT labels (expected only for video_1)
    gt_dir = video_dir / "gt"
    if gt_dir.exists():
        gt_file = gt_dir / "gt.txt"
        if gt_file.exists():
            result["has_gt"] = True
        else:
            result["issues"].append("gt/ directory exists but gt/gt.txt is missing")

    # seqinfo.ini (optional but useful for TrackEval)
    seqinfo = video_dir / "seqinfo.ini"
    if not seqinfo.exists():
        result["issues"].append("seqinfo.ini missing (optional, needed for TrackEval FPS info)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check lab_data directory structure for all 5 tracking videos"
    )
    parser.add_argument(
        "--lab-data-root",
        type=str,
        default=os.environ.get("LAB_DATA", ""),
        help="Path to folder containing video_1 ... video_5. "
             "Defaults to LAB_DATA environment variable.",
    )
    args = parser.parse_args()

    if not args.lab_data_root:
        print("ERROR: --lab-data-root not set and LAB_DATA environment variable is empty.")
        print("  Example: python scripts/check_data.py --lab-data-root C:/path/to/lab_data")
        sys.exit(1)

    root = Path(args.lab_data_root).resolve()
    print(f"\n{'='*60}")
    print(f"  Checking lab data at: {root}")
    print(f"{'='*60}\n")

    if not root.exists():
        print(f"ERROR: Root directory does not exist: {root}")
        print("  Check the path and make sure the data archive was extracted correctly.")
        sys.exit(1)

    all_ok = True
    video_names = [f"video_{i}" for i in range(1, 6)]

    for name in video_names:
        video_dir = root / name
        res = check_video(video_dir, name)

        ok = res["exists"] and res["img1_exists"] and res["img_count"] > 0
        status_icon = "OK  " if ok else "FAIL"
        gt_info = "Has GT labels (use TrackEval for HOTA/MOTA/IDF1)" if res["has_gt"] else "No GT labels (evaluate visually)"

        print(f"[{status_icon}] {name}:")
        print(f"       Images in img1/ : {res['img_count']}")
        print(f"       Labels          : {gt_info}")

        for issue in res["issues"]:
            if "seqinfo" in issue:
                print(f"       WARNING: {issue}")
            else:
                all_ok = False
                print(f"       ERROR:   {issue}")
        print()

    print("-" * 60)
    if all_ok:
        print("All data looks good. Ready to run tracking!\n")
    else:
        print("Some issues found above. Fix them before running tracking.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
