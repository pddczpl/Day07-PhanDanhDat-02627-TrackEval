# -*- coding: utf-8 -*-
"""
eval_video1.py - Evaluate tracking results on video_1 using TrackEval metrics.

Computes and prints:
- HOTA, DetA, AssA
- MOTA, MOTP, IDSW, FP, FN
- IDF1, IDP, IDR

Usage:
    python scripts/eval_video1.py --pred runs/nop_bai/video_1.txt
    python scripts/eval_video1.py --pred runs/nop_bai/video_1.txt --gt data_lab/video_1/gt/gt.txt
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trackeval
from trackeval.metrics import HOTA, CLEAR, Identity


def load_mot_file(filepath: Path, is_gt: bool = False):
    """
    Load a MOT-format txt file into a dictionary of frame -> detections.
    Each detection: [id, x1, y1, w, h, conf, (class, visibility)]
    """
    data_by_frame = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                parts = line.strip().split()
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0

            # GT filtering in MOT17:
            # col 7: class_id (1: pedestrian, 2: person on vehicle, 7: static person, ...)
            # col 8: visibility (0.0 to 1.0)
            if is_gt and len(parts) >= 8:
                class_id = int(parts[7])
                # Only keep pedestrian class (class 1)
                if class_id != 1:
                    continue

            if frame_id not in data_by_frame:
                data_by_frame[frame_id] = []

            # Format: [x1, y1, x2, y2, id]
            data_by_frame[frame_id].append([x, y, x + w, y + h, track_id])

    return data_by_frame


def evaluate(pred_file: str, gt_file: str):
    pred_path = Path(pred_file).resolve()
    gt_path = Path(gt_file).resolve()

    if not pred_path.exists():
        print(f"ERROR: Prediction file not found: {pred_path}")
        sys.exit(1)
    if not gt_path.exists():
        print(f"ERROR: Ground truth file not found: {gt_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Evaluating video_1:")
    print(f"  Prediction : {pred_path.name}")
    print(f"  Groundtruth: {gt_path}")
    print(f"{'='*60}\n")

    gt_data = load_mot_file(gt_path, is_gt=True)
    pred_data = load_mot_file(pred_path, is_gt=False)

    # Determine frame range
    all_frames = sorted(set(gt_data.keys()).union(set(pred_data.keys())))
    num_timesteps = max(all_frames) if all_frames else 0

    # Convert to trackeval input format:
    # data = {
    #   'gt_ids': list of np.array(N_t) per timestep,
    #   'gt_dets': list of np.array(N_t, 4) [x1, y1, x2, y2] per timestep,
    #   'tracker_ids': list of np.array(M_t) per timestep,
    #   'tracker_dets': list of np.array(M_t, 4) per timestep,
    #   'similarity_scores': list of np.array(N_t, M_t) per timestep,
    #   'num_tracker_dets': total,
    #   'num_gt_dets': total,
    #   'num_tracker_ids': total unique,
    #   'num_gt_ids': total unique,
    #   'num_timesteps': num_timesteps
    # }

    gt_ids_list = []
    gt_dets_list = []
    tracker_ids_list = []
    tracker_dets_list = []
    sim_scores_list = []

    unique_gt_ids = set()
    unique_tr_ids = set()
    total_gt_dets = 0
    total_tr_dets = 0

    for t in range(1, num_timesteps + 1):
        # GT
        gts = gt_data.get(t, [])
        if gts:
            gts_arr = np.array(gts, dtype=np.float32)
            g_boxes = gts_arr[:, :4]
            g_ids = gts_arr[:, 4].astype(int)
            unique_gt_ids.update(g_ids.tolist())
            total_gt_dets += len(g_ids)
        else:
            g_boxes = np.empty((0, 4), dtype=np.float32)
            g_ids = np.empty((0,), dtype=int)

        # Tracker
        trs = pred_data.get(t, [])
        if trs:
            trs_arr = np.array(trs, dtype=np.float32)
            t_boxes = trs_arr[:, :4]
            t_ids = trs_arr[:, 4].astype(int)
            unique_tr_ids.update(t_ids.tolist())
            total_tr_dets += len(t_ids)
        else:
            t_boxes = np.empty((0, 4), dtype=np.float32)
            t_ids = np.empty((0,), dtype=int)

        # Compute IoU matrix for timestep
        if len(g_boxes) > 0 and len(t_boxes) > 0:
            # pairwise IoU
            x1 = np.maximum(g_boxes[:, None, 0], t_boxes[None, :, 0])
            y1 = np.maximum(g_boxes[:, None, 1], t_boxes[None, :, 1])
            x2 = np.minimum(g_boxes[:, None, 2], t_boxes[None, :, 2])
            y2 = np.minimum(g_boxes[:, None, 3], t_boxes[None, :, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_g = (g_boxes[:, 2] - g_boxes[:, 0]) * (g_boxes[:, 3] - g_boxes[:, 1])
            area_t = (t_boxes[:, 2] - t_boxes[:, 0]) * (t_boxes[:, 3] - t_boxes[:, 1])
            union = area_g[:, None] + area_t[None, :] - inter
            sim = np.where(union > 0, inter / union, 0.0)
        else:
            sim = np.empty((len(g_boxes), len(t_boxes)), dtype=np.float32)

        gt_ids_list.append(g_ids)
        gt_dets_list.append(g_boxes)
        tracker_ids_list.append(t_ids)
        tracker_dets_list.append(t_boxes)
        sim_scores_list.append(sim)

    # Re-label IDs to be contiguous 0..N-1 for HOTA index matrix
    if len(unique_gt_ids) > 0:
        sorted_gt = np.sort(list(unique_gt_ids))
        gt_map = {val: idx for idx, val in enumerate(sorted_gt)}
        for t in range(num_timesteps):
            if len(gt_ids_list[t]) > 0:
                gt_ids_list[t] = np.array([gt_map[i] for i in gt_ids_list[t]], dtype=int)

    if len(unique_tr_ids) > 0:
        sorted_tr = np.sort(list(unique_tr_ids))
        tr_map = {val: idx for idx, val in enumerate(sorted_tr)}
        for t in range(num_timesteps):
            if len(tracker_ids_list[t]) > 0:
                tracker_ids_list[t] = np.array([tr_map[i] for i in tracker_ids_list[t]], dtype=int)

    eval_data = {
        "gt_ids": gt_ids_list,
        "gt_dets": gt_dets_list,
        "tracker_ids": tracker_ids_list,
        "tracker_dets": tracker_dets_list,
        "similarity_scores": sim_scores_list,
        "num_tracker_dets": total_tr_dets,
        "num_gt_dets": total_gt_dets,
        "num_tracker_ids": len(unique_tr_ids),
        "num_gt_ids": len(unique_gt_ids),
        "num_timesteps": num_timesteps,
    }

    # Evaluate metrics
    hota_metric = HOTA()
    clear_metric = CLEAR()
    id_metric = Identity()

    res_hota = hota_metric.eval_sequence(eval_data)
    res_clear = clear_metric.eval_sequence(eval_data)
    res_id = id_metric.eval_sequence(eval_data)

    hota_val = res_hota["HOTA"][0] * 100
    deta_val = res_hota["DetA"][0] * 100
    assa_val = res_hota["AssA"][0] * 100

    mota_val = res_clear["MOTA"] * 100
    idsw_val = int(res_clear["IDSW"])
    fp_val = int(res_clear["CLR_FP"])
    fn_val = int(res_clear["CLR_FN"])

    idf1_val = res_id["IDF1"] * 100
    idp_val = res_id["IDP"] * 100
    idr_val = res_id["IDR"] * 100

    print(f"+------------------------------------------------------------+")
    print(f"|                     EVALUATION RESULTS                     |")
    print(f"+------------------------------------------------------------+")
    print(f"|  Metric    |  Score (%)  |  Detailed Sub-metrics           |")
    print(f"+------------+-------------+---------------------------------+")
    print(f"|  HOTA      |   {hota_val:6.2f}%   |  DetA: {deta_val:5.2f}% | AssA: {assa_val:5.2f}%   |")
    print(f"|  MOTA      |   {mota_val:6.2f}%   |  FP: {fp_val:5d} | FN: {fn_val:5d} | IDSW: {idsw_val:3d} |")
    print(f"|  IDF1      |   {idf1_val:6.2f}%   |  IDP: {idp_val:5.2f}% | IDR: {idr_val:5.2f}%    |")
    print(f"+------------+-------------+---------------------------------+")
    print()
    print("Ghi chu cho bao cao:")
    print(f"- HOTA : {hota_val:.2f}% (Can bang giua Detection va Association)")
    print(f"- MOTA : {mota_val:.2f}% (Anh huong boi FP: {fp_val}, FN: {fn_val}, IDSW: {idsw_val})")
    print(f"- IDF1 : {idf1_val:.2f}% (Do on dinh giu danh tinh theo thoi gian)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate video_1 using TrackEval")
    parser.add_argument(
        "--pred",
        default="runs/nop_bai/video_1.txt",
        help="Path to prediction txt (default: runs/nop_bai/video_1.txt)",
    )
    parser.add_argument(
        "--gt",
        default="data_lab/video_1/gt/gt.txt",
        help="Path to groundtruth txt (default: data_lab/video_1/gt/gt.txt)",
    )
    args = parser.parse_args()
    evaluate(args.pred, args.gt)


if __name__ == "__main__":
    main()
