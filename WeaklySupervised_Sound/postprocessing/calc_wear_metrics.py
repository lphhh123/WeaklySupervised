import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import compute_misalignment_measures, ANETdetection, convert_segments_to_samples
import warnings


def warn(*args, **kwargs):
    pass



warnings.warn = warn

PRED_PATH = '/home/yinjiaxi/wstal/tal_for_har/logs/actionformer/2025-12-03 12:45:25_/'


ANNO_PATH = '/home/yinjiaxi/wstal/tal_for_har/data/wear/annotations/'

RAW_DATA_PATH = '/home/yinjiaxi/wstal/tal_for_har/data/wear/raw/inertial'

DATASET = 'wear'
NUM_CLASSES = 19
SAMPLING_RATE = 50
INPUT_DIM = 12
SUBJECT_IDS = range(18)

SCORE_THRES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


# ===========================================

def main():
    print(f"Starting evaluation for dataset: {DATASET}")
    print(f"Prediction path: {PRED_PATH}")
    print(f"Annotation path: {ANNO_PATH}")
    print("-" * 60)

    header = "{:<10} | {:<8} {:<8} {:<8} | {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} | {:<8}".format(
        "Threshold", "P", "R", "F1", "UR", "OR", "DR", "IR", "FR", "MR", "mAP"
    )
    print(header)
    print("-" * 110)

    for f in SCORE_THRES:
        all_mAP = []
        all_ur = np.zeros(NUM_CLASSES - 1)
        all_dr = np.zeros(NUM_CLASSES - 1)
        all_fr = np.zeros(NUM_CLASSES - 1)
        all_ir = np.zeros(NUM_CLASSES - 1)
        all_or = np.zeros(NUM_CLASSES - 1)
        all_mr = np.zeros(NUM_CLASSES - 1)

        all_prec = np.zeros(NUM_CLASSES)
        all_recall = np.zeros(NUM_CLASSES)
        all_f1 = np.zeros(NUM_CLASSES)

        valid_subject_count = 0

        for i in SUBJECT_IDS:
            sbj_name = f'loso_sbj_{i}'
            json_file_path = os.path.join(ANNO_PATH, f'{sbj_name}.json')
            pred_file_path = os.path.join(PRED_PATH, f'v_seg_{sbj_name}.csv')

            if not os.path.exists(json_file_path):
                print(f"Warning: Annotation file not found: {json_file_path}")
                continue
            if not os.path.exists(pred_file_path):
                print(f"Warning: Prediction file not found: {pred_file_path}")
                continue

            valid_subject_count += 1

            with open(json_file_path) as fi:
                file = json.load(fi)
            anno_db = file['database']
            labels = ['null'] + list(file['label_dict'])
            label_dict = dict(zip(labels, list(range(len(labels)))))

            v_seg = pd.read_csv(pred_file_path, index_col=None, low_memory=False)

            v_data = np.empty((0, INPUT_DIM + 2))
            val_sbjs = [x for x in anno_db if anno_db[x]['subset'] == 'Validation']

            for sbj in val_sbjs:
                raw_csv_path = os.path.join(RAW_DATA_PATH, sbj + '.csv')
                if not os.path.exists(raw_csv_path):
                    print(f"Error: Raw data not found: {raw_csv_path}")
                    continue
                data = pd.read_csv(raw_csv_path, index_col=False, low_memory=False).replace(
                    {"label": label_dict}).fillna(0).to_numpy()
                v_data = np.append(v_data, data, axis=0)

            v_seg_filtered = v_seg[v_seg.score > f].copy()
            v_seg_filtered = v_seg_filtered.rename(
                columns={"video_id": "video-id", "t_start": "t-start", "t_end": "t-end"})

            det_eval = ANETdetection(json_file_path, 'validation', tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                                     verbose=False)
            v_mAP, _ = det_eval.evaluate(v_seg_filtered)
            all_mAP.append(v_mAP)

            preds, gt, _ = convert_segments_to_samples(v_seg_filtered, v_data, SAMPLING_RATE, threshold=f)

            eval_labels = range(NUM_CLASSES)

            v_prec = precision_score(gt, preds, average=None, labels=eval_labels, zero_division=0)
            v_rec = recall_score(gt, preds, average=None, labels=eval_labels, zero_division=0)
            v_f1_score = f1_score(gt, preds, average=None, labels=eval_labels, zero_division=0)

            all_prec += v_prec
            all_recall += v_rec
            all_f1 += v_f1_score

            v_ur, v_dr, v_fr, v_ir, v_or, v_mr = compute_misalignment_measures(gt, preds, eval_labels)
            all_ur += v_ur
            all_dr += v_dr
            all_fr += v_fr
            all_ir += v_ir
            all_or += v_or
            all_mr += v_mr

        if valid_subject_count == 0:
            print("No valid files found.")
            break

        avg_mAP = np.mean(all_mAP) * 100

        avg_prec = np.mean(all_prec) / valid_subject_count * 100
        avg_rec = np.mean(all_recall) / valid_subject_count * 100
        avg_f1 = np.mean(all_f1) / valid_subject_count * 100

        avg_ur = np.mean(all_ur) / valid_subject_count * 100
        avg_or = np.mean(all_or) / valid_subject_count * 100
        avg_dr = np.mean(all_dr) / valid_subject_count * 100
        avg_ir = np.mean(all_ir) / valid_subject_count * 100
        avg_fr = np.mean(all_fr) / valid_subject_count * 100
        avg_mr = np.mean(all_mr) / valid_subject_count * 100

        print(
            "{:<10} | {:<8.2f} {:<8.2f} {:<8.2f} | {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} | {:<8.2f}".format(
                f, avg_prec, avg_rec, avg_f1, avg_ur, avg_or, avg_dr, avg_ir, avg_fr, avg_mr, avg_mAP
            ))


if __name__ == '__main__':
    main()
