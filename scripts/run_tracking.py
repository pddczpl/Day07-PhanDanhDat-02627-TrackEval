# -*- coding: utf-8 -*-
"""
run_tracking.py - Pipeline: YOLO detector -> Tracker -> video_N.txt + preview video.

Usage:
    python scripts/run_tracking.py ^
        --source %LAB_DATA%/video_1/img1 ^
        --seq-name video_1 ^
        --tracker bytetrack ^
        --conf 0.3 --iou 0.5 ^
        --out runs/thu_nhanh ^
        --save-video --max-frames 150

Trackers: bytetrack, ocsort, botsort, strongsort, deepocsort
YOLO model: yolo11n.pt (auto-downloaded on first run)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ─── Stable color per track ID ─────────────────────────────────────────────────
def id_to_color(track_id: int) -> tuple:
    """Map track ID to a stable BGR color."""
    np.random.seed(track_id * 31 + 7)
    return tuple(int(c) for c in np.random.randint(50, 230, 3))


# ─── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Lab07 tracking pipeline: YOLO detection -> BoxMOT tracker -> MOT .txt output"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to img1/ folder of the video sequence (e.g. lab_data/video_1/img1)",
    )
    parser.add_argument(
        "--seq-name",
        required=True,
        help="Sequence name used for naming output files (e.g. video_1)",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack",
        choices=["bytetrack", "ocsort", "botsort", "strongsort", "deepocsort"],
        help="Tracker algorithm to use (default: bytetrack)",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Path to YOLO model weights (default: yolo11n.pt; auto-downloads if missing)",
    )
    parser.add_argument(
        "--reid-model",
        default="osnet_x0_25_msmt17.pt",
        help="Re-ID model path for botsort/strongsort/deepocsort",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="YOLO confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold for YOLO detector (default: 0.5). "
             "NOTE: this is detector NMS, NOT the tracker association threshold.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="YOLO inference image size in pixels (default: 640)",
    )
    import torch
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser.add_argument(
        "--device",
        default=default_device,
        help=f"Inference device: 'cpu' or 'cuda:0' (default: {default_device})",
    )
    parser.add_argument(
        "--out",
        default="runs/tracking",
        help="Output directory for .txt and preview video (default: runs/tracking)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save a preview MP4 video with colored bounding boxes and track IDs",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process. QUICK TEST ONLY — remove this for final submission!",
    )

    return parser.parse_args()


# ─── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Late imports to give clear error messages if packages missing
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not found. Run: pip install ultralytics")
        sys.exit(1)

    try:
        from boxmot import ByteTrack, OcSort, BotSort, StrongSort, DeepOcSort
        TRACKER_MAP = {
            "bytetrack": ByteTrack,
            "ocsort": OcSort,
            "botsort": BotSort,
            "strongsort": StrongSort,
            "deepocsort": DeepOcSort,
        }
    except ImportError:
        print("ERROR: boxmot not found. Run: pip install boxmot")
        sys.exit(1)

    # ── Prepare paths ───────────────────────────────────────────────
    source_dir = Path(args.source).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print(f"ERROR: Image directory not found: {source_dir}")
        sys.exit(1)

    # Sort image files by filename
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted(
        [f for f in source_dir.iterdir() if f.suffix.lower() in img_exts],
        key=lambda x: x.name,
    )

    if not img_files:
        print(f"ERROR: No images found in: {source_dir}")
        sys.exit(1)

    if args.max_frames:
        img_files = img_files[: args.max_frames]
        print(f"[QUICK TEST] Limited to first {args.max_frames} frames.")

    print(f"\n{'='*60}")
    print(f"  Tracker  : {args.tracker.upper()}")
    print(f"  Model    : {args.model}")
    print(f"  Frames   : {len(img_files)}")
    print(f"  conf={args.conf}  |  iou={args.iou}  |  device={args.device}")
    print(f"  Output   : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load YOLO model ──────────────────────────────────────────────
    print(f"[1/3] Loading YOLO model: {args.model} ...")
    model = YOLO(args.model)

    # ── Initialize tracker ───────────────────────────────────────────
    print(f"[2/3] Initializing tracker: {args.tracker} ...")
    TrackerClass = TRACKER_MAP[args.tracker]

    tracker_kwargs = {"per_class": False}

    # Re-ID trackers need reid_weights and device
    if args.tracker in ("botsort", "strongsort", "deepocsort"):
        reid_path = Path(args.reid_model)
        if not reid_path.exists():
            print(f"  WARNING: Re-ID model not found at {reid_path}")
            print("  BoxMOT will auto-download osnet_x0_25_msmt17.pt on first run.")
        tracker_kwargs["reid_weights"] = reid_path
        tracker_kwargs["device"] = args.device

    tracker = TrackerClass(**tracker_kwargs)

    # ── Setup video writer ───────────────────────────────────────────
    video_writer = None
    preview_path = out_dir / f"{args.seq_name}_preview.mp4"
    if args.save_video:
        first_img = cv2.imread(str(img_files[0]))
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(preview_path), fourcc, 10, (w, h))
        print(f"  Video preview will be saved to: {preview_path}")

    # ── MOT format output ────────────────────────────────────────────
    txt_path = out_dir / f"{args.seq_name}.txt"
    mot_lines = []

    # ── Frame loop ───────────────────────────────────────────────────
    print(f"[3/3] Running detection + tracking on {len(img_files)} frames ...")
    t0 = time.time()

    for frame_idx, img_path in enumerate(img_files, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: Could not read image: {img_path}")
            continue

        # YOLO detect — class 0 = person only
        results = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            classes=[0],          # person class
            imgsz=args.img_size,
            device=args.device,
            verbose=False,
        )

        # Build detection array: [[x1, y1, x2, y2, conf, cls], ...]
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            dets = results[0].boxes.data.cpu().numpy()   # shape (N, 6)
        else:
            dets = np.empty((0, 6))

        # Tracker update
        # Input:  (N, 6) [x1, y1, x2, y2, conf, cls]
        # Output: (M, 7) [x1, y1, x2, y2, track_id, conf, cls]
        tracks = tracker.update(dets, frame)

        # Write MOT lines and draw on frame
        if len(tracks) > 0:
            for t in tracks:
                x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                track_id = int(t[4])
                conf_score = float(t[5]) if len(t) > 5 else -1.0

                # MOT format: frame, id, left, top, width, height, conf, -1, -1, -1
                w_box = x2 - x1
                h_box = y2 - y1
                mot_lines.append(
                    f"{frame_idx},{track_id},{x1},{y1},{w_box},{h_box},{conf_score:.4f},-1,-1,-1"
                )

                if video_writer is not None:
                    color = id_to_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(
                        frame, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                    )

        if video_writer is not None:
            info = f"Frame:{frame_idx:04d} | {args.tracker.upper()} | conf:{args.conf} iou:{args.iou}"
            cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            video_writer.write(frame)

        # Progress log every 50 frames
        if frame_idx % 50 == 0 or frame_idx == len(img_files):
            elapsed = time.time() - t0
            fps = frame_idx / elapsed
            print(f"  Frame {frame_idx:4d}/{len(img_files)} | {fps:.1f} FPS | tracks: {len(tracks)}")

    # ── Save output ──────────────────────────────────────────────────
    with open(txt_path, "w") as f:
        f.write("\n".join(mot_lines))

    if video_writer is not None:
        video_writer.release()

    elapsed_total = time.time() - t0
    print(f"\nDone! {len(img_files)} frames in {elapsed_total:.1f}s")
    print(f"  MOT result : {txt_path}")
    if args.save_video:
        print(f"  Preview    : {preview_path}")
    print()


if __name__ == "__main__":
    main()
