import json
from pathlib import Path

import numpy as np


def load_results(json_path_or_obj):
    if isinstance(json_path_or_obj, (str, Path)):
        with open(json_path_or_obj, "r", encoding="utf-8") as f:
            return json.load(f)
    return json_path_or_obj


def summarize_map_across_folds(results, splits=("test_window", "test_full")):
    out = {}

    for split in splits:

        avg_maps = []
        per_tiou_maps = []
        tious_ref = None

        for r in results:
            if split not in r:
                continue
            s = r[split]
            avg_maps.append(float(s["avg_mAP"]))
            per_tiou_maps.append([float(x) for x in s["mAPs"]])

            if tious_ref is None:
                tious_ref = [float(x) for x in s["tious"]]

        if not avg_maps:
            out[split] = None
            continue

        avg_maps_np = np.array(avg_maps, dtype=np.float64)
        per_tiou_np = np.array(per_tiou_maps, dtype=np.float64)

        out[split] = {
            "num_folds": int(len(avg_maps)),
            "mean_avg_mAP": float(avg_maps_np.mean()),
            "std_avg_mAP": float(avg_maps_np.std(ddof=1)) if len(avg_maps_np) > 1 else 0.0,
            "tious": tious_ref,
            "mean_mAPs_per_tiou": per_tiou_np.mean(axis=0).tolist(),
        }

    return out


def pretty_print_summary(summary):
    for split, s in summary.items():
        if s is None:
            print(f"[{split}] no data")
            continue

        print(f"\n[{split}] folds={s['num_folds']}")
        print(f"  mean(avg_mAP) = {s['mean_avg_mAP']:.6f}   std = {s['std_avg_mAP']:.6f}")
        print("  mean mAP @ tIoU:")
        for tiou, m in zip(s["tious"], s["mean_mAPs_per_tiou"]):
            print(f"    tIoU={tiou:.1f}: {m:.6f}")


if __name__ == "__main__":
    results = load_results("/home/lipei/project/WSDDN/test_results/HANGTIME/pcl_0108/loso_report_partial.json")

    summary = summarize_map_across_folds(results, splits=("test_window", "test_full"))
    pretty_print_summary(summary)

