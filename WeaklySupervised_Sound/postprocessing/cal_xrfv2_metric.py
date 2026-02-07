# -*- coding: utf-8 -*-
import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import compute_misalignment_measures, ANETdetection, convert_segments_to_samples
import warnings


def warn(*args, **kwargs): pass


warnings.warn = warn


RESULT_BASE = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_cdur/"
PRED_PATHS = {
    "window": os.path.join(RESULT_BASE, "pred_window.json"),
    "full": os.path.join(RESULT_BASE, "prediction_test_full.json")
}
GT_PATH = "/home/lipei/XRFV2/imu_annotations.json"
MAPPING_PATH = "/home/lipei/project/WSDDN/label_mapping.json"
RESULT_SAVE_DIR = "/home/yinjiaxi/wstal/WeaklySupervised-master/metric_result/cdur/"

DATASET = 'xrfv2'
SEED = '2022'
NUM_CLASSES = 30
SAMPLING_RATE = 50
INPUT_DIM = 36
SCORE_THRES = [0.0]


# ===========================================

def load_mapping_and_fix_gt(gt_path, mapping_path):
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    name_to_id = {v: int(k) for k, v in mapping['new_id_to_action'].items()}

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    new_db = {}
    for vid, info in gt_data['database'].items():
        shape_str = info.get('data_shape', '(0,)')
        match = re.search(r'\((\d+)', shape_str)
        num_samples = int(match.group(1)) if match else 0
        duration = num_samples / SAMPLING_RATE

        valid_anns = []
        for ann in info['annotations']:
            raw_label = str(ann['label'])
            if raw_label in name_to_id:
                label_id = name_to_id[raw_label]
            elif "Walking" in raw_label:
                label_id = 24
            else:
                continue

            s_sec = ann['segment'][0] / SAMPLING_RATE
            e_sec = ann['segment'][1] / SAMPLING_RATE

            valid_anns.append({
                "segment": [s_sec, e_sec],
                "label": str(label_id),
                "label_id": label_id
            })
        new_db[vid] = {"duration": duration, "annotations": valid_anns}

    gt_data['database'] = new_db
    return "/tmp/xrfv2_gt_temp.json", gt_data, mapping['new_id_to_action']


def evaluate_xrfv2(mode_suffix, fixed_gt_path, gt_raw_data, action_map):
    pred_path = PRED_PATHS[mode_suffix]
    if not os.path.exists(pred_path):
        print(f"Warning: {pred_path} not found.")
        return None

    with open(pred_path, 'r') as f_p:
        pred_data = json.load(f_p)
    results_dict = pred_data.get('results', pred_data)



    rev_action_map = {str(v).strip(): int(k) for k, v in action_map.items()}

    print(f"\n{'=' * 20} Evaluating XRFV2: {mode_suffix.upper()} {'=' * 20}")

    for f in SCORE_THRES:
        all_preds, all_gts = [], []
        for vid, info in gt_raw_data['database'].items():
            duration = info['duration']
            total_samples = int(duration * SAMPLING_RATE)

            v_data = np.zeros((total_samples, INPUT_DIM + 2))

            for ann in info['annotations']:
                s_idx = max(0, int(ann['segment'][0] * SAMPLING_RATE))
                e_idx = min(total_samples, int(ann['segment'][1] * SAMPLING_RATE))
                v_data[s_idx:e_idx, -1] = ann['label_id']

            if vid in results_dict:
                v_df = pd.DataFrame(results_dict[vid], columns=['t-start', 't-end', 'label', 'score'])
                v_df['video-id'] = vid



                v_df['label'] = v_df['label'].astype(str).str.strip()

                v_df['label_id'] = v_df['label'].map(lambda x: rev_action_map.get(x, -1))


                v_df = v_df[(v_df['score'] > f) & (v_df['label_id'] != -1)].copy()


                v_df['label'] = v_df['label_id'].astype(int)

                v_preds_samples, _, _ = convert_segments_to_samples(v_df, v_data, SAMPLING_RATE, threshold=f)
            else:
                v_preds_samples = np.zeros(total_samples)

            all_preds.extend(v_preds_samples)
            all_gts.extend(v_data[:, -1])

        all_preds, all_gts = np.array(all_preds), np.array(all_gts)
        eval_range = range(NUM_CLASSES)


        prec = precision_score(all_gts, all_preds, average='macro', labels=list(eval_range), zero_division=0) * 100
        rec = recall_score(all_gts, all_preds, average='macro', labels=list(eval_range), zero_division=0) * 100
        f1 = f1_score(all_gts, all_preds, average='macro', labels=list(eval_range), zero_division=0) * 100
        ur, dr, fr, ir, or_, mr = compute_misalignment_measures(all_gts, all_preds, eval_range)

        print("{:<10} | {:<8.2f} {:<8.2f} {:<8.2f} | {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.4f}".format(
            f, prec, rec, f1, ur.mean() * 100, or_.mean() * 100, dr.mean() * 100, ir.mean() * 100, fr.mean() * 100,
                              mr.mean() * 100
        ))

        return {"threshold": f, "precision": float(prec), "recall": float(rec), "f1": float(f1),
                "ur": float(ur.mean() * 100), "or": float(or_.mean() * 100), "dr": float(dr.mean() * 100),
                "ir": float(ir.mean() * 100), "fr": float(fr.mean() * 100), "mr": float(mr.mean() * 100)}


def main():
    fixed_gt_path, gt_raw_data, action_map = load_mapping_and_fix_gt(GT_PATH, MAPPING_PATH)

    window_res = evaluate_xrfv2("window", fixed_gt_path, gt_raw_data, action_map)
    full_res = evaluate_xrfv2("full", fixed_gt_path, gt_raw_data, action_map)

    final_json = {
        "dataset": DATASET,
        "results": {
            "window": [window_res] if window_res else [],
            "full": [full_res] if full_res else []
        }
    }

    os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_SAVE_DIR, f"{DATASET}_{SEED}.json")
    with open(save_path, 'w') as f:
        json.dump(final_json, f, indent=4)
    print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    main()