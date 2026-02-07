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



RESULT_ROOT_PATH = "/home/yinjiaxi/wstal/WeaklySupervised-master/result//sbhar_cdur_10_15s_2022/"
RAW_DATA_PATH = '/home/yinjiaxi/wstal/tal_for_har/data/sbhar/raw/inertial'
SEED = '2022'
DATASET = 'sbhar'
NUM_CLASSES = 13
SAMPLING_RATE = 50
INPUT_DIM = 3
FOLDS_IDS = range(30)


RESULT_SAVE_DIR = "/home/yinjiaxi/wstal/WeaklySupervised-master/metric_result/cdur/"


print("dataset: sbhar")
print("model: cdur")


SCORE_THRES = [0.0]


# ===========================================

def build_label_dict(gt_db):
    unique_labels = set()
    for sbj in gt_db:
        for ann in gt_db[sbj]['annotations']:
            unique_labels.add(ann['label'])
    sorted_labels = sorted(list(unique_labels))
    labels_list = ['null'] + sorted_labels
    label_dict = {label: idx for idx, label in enumerate(labels_list)}
    return label_dict, labels_list


def evaluate_loso_folds(mode_suffix="window"):
    """
    Message：Message，Message
    """
    task_name = f"Predictions: {mode_suffix.upper()}"
    print(f"\n{'=' * 20} Running LOSO Evaluation for: {task_name} {'=' * 20}")

    print("-" * 110)
    header = "{:<10} | {:<8} {:<8} {:<8} | {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} | {:<8}".format(
        "Threshold", "P", "R", "F1", "UR", "OR", "DR", "IR", "FR", "MR", "mAP"
    )
    print(header)
    print("-" * 110)


    metrics_list = []

    for f in SCORE_THRES:
        all_mAP = []

        all_ur = None
        all_dr = None
        all_fr = None
        all_ir = None
        all_or = None
        all_mr = None
        all_prec = None
        all_recall = None
        all_f1 = None

        valid_fold_count = 0

        for fold_idx in FOLDS_IDS:
            fold_name = f"fold{fold_idx}"
            sbj_name = f"sbj_{fold_idx}"

            fold_dir = os.path.join(RESULT_ROOT_PATH, fold_name)
            pred_path = os.path.join(fold_dir, f'best_predictions_test_{mode_suffix}_15s.json')
            gt_path = os.path.join(fold_dir, 'gt_for_anet.json')

            if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                continue

            with open(pred_path, 'r') as f_p:
                pred_data = json.load(f_p)
            with open(gt_path, 'r') as f_g:
                gt_data = json.load(f_g)
            gt_db = gt_data['database']

            label_dict, labels_list = build_label_dict(gt_db)
            num_labels = len(labels_list)

            if all_ur is None:
                metric_dim = max(1, num_labels - 1)
                all_ur = np.zeros(metric_dim)
                all_dr = np.zeros(metric_dim)
                all_fr = np.zeros(metric_dim)
                all_ir = np.zeros(metric_dim)
                all_or = np.zeros(metric_dim)
                all_mr = np.zeros(metric_dim)
                all_prec = np.zeros(num_labels)
                all_recall = np.zeros(num_labels)
                all_f1 = np.zeros(num_labels)


            if sbj_name not in pred_data['results']:
                v_seg = pd.DataFrame(columns=['video-id', 't-start', 't-end', 'label', 'score'])
            else:
                curr_pred_list = pred_data['results'][sbj_name]
                df_data = []
                for item in curr_pred_list:
                    df_data.append({
                        'video-id': sbj_name,
                        't-start': item['segment'][0],
                        't-end': item['segment'][1],
                        'label': item['label'],
                        'score': item['score']
                    })
                v_seg = pd.DataFrame(df_data)


            raw_csv_path = os.path.join(RAW_DATA_PATH, f"loso_{sbj_name}.csv")
            if not os.path.exists(raw_csv_path):
                raw_csv_path = os.path.join(RAW_DATA_PATH, f"{sbj_name}.csv")
                if not os.path.exists(raw_csv_path):
                    continue

            v_data = np.empty((0, INPUT_DIM + 2))
            data_df = pd.read_csv(raw_csv_path, index_col=False, low_memory=False)
            data = data_df.replace({"label": label_dict}).fillna(0).to_numpy()
            v_data = np.append(v_data, data, axis=0)

            valid_fold_count += 1

            if not v_seg.empty:
                v_seg_filtered = v_seg[v_seg.score > f].copy()
            else:
                v_seg_filtered = v_seg.copy()

            v_seg_filtered = v_seg_filtered.rename(
                columns={"video_id": "video-id", "t_start": "t-start", "t_end": "t-end"})


            det_eval = ANETdetection(gt_path, 'test', tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])
            v_mAP, _ = det_eval.evaluate(v_seg_filtered)
            all_mAP.append(v_mAP)


            v_seg_for_samples = v_seg_filtered.copy()
            v_seg_for_samples['label'] = v_seg_for_samples['label'].map(label_dict) - 1
            v_seg_for_samples = v_seg_for_samples.dropna(subset=['label'])

            preds, gt, _ = convert_segments_to_samples(v_seg_for_samples, v_data, SAMPLING_RATE, threshold=f)
            eval_labels = range(num_labels)

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

        if valid_fold_count == 0:
            print(f"No valid folds found for threshold {f}")
            continue


        avg_mAP = np.mean(all_mAP) * 100
        avg_prec = np.mean(all_prec) / valid_fold_count * 100
        avg_rec = np.mean(all_recall) / valid_fold_count * 100
        avg_f1 = np.mean(all_f1) / valid_fold_count * 100
        avg_ur = np.mean(all_ur) / valid_fold_count * 100
        avg_or = np.mean(all_or) / valid_fold_count * 100
        avg_dr = np.mean(all_dr) / valid_fold_count * 100
        avg_ir = np.mean(all_ir) / valid_fold_count * 100
        avg_fr = np.mean(all_fr) / valid_fold_count * 100
        avg_mr = np.mean(all_mr) / valid_fold_count * 100

        print(
            "{:<10} | {:<8.2f} {:<8.2f} {:<8.2f} | {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.4f} | {:<8.2f}".format(
                f, avg_prec, avg_rec, avg_f1, avg_ur, avg_or, avg_dr, avg_ir, avg_fr, avg_mr, avg_mAP
            ))


        metrics_list.append({
            "threshold": f,
            "precision": float(avg_prec),
            "recall": float(avg_rec),
            "f1": float(avg_f1),
            "ur": float(avg_ur),
            "or": float(avg_or),
            "dr": float(avg_dr),
            "ir": float(avg_ir),
            "fr": float(avg_fr),
            "mr": float(avg_mr),
            "mAP": float(avg_mAP)
        })

    return metrics_list


def main():

    window_results = evaluate_loso_folds(mode_suffix="window")

    print("\n" * 2)


    full_results = evaluate_loso_folds(mode_suffix="full")


    final_json_data = {
        "dataset": DATASET,
        "seed": SEED,
        "model_type": "cdur/cdur (Weakly Supervised)",
        "results": {
            "window": window_results,
            "full": full_results
        }
    }


    if not os.path.exists(RESULT_SAVE_DIR):
        os.makedirs(RESULT_SAVE_DIR)
        print(f"Created directory: {RESULT_SAVE_DIR}")

    json_filename = f"{DATASET}_{SEED}.json"
    save_path = os.path.join(RESULT_SAVE_DIR, json_filename)

    with open(save_path, 'w') as f:
        json.dump(final_json_data, f, indent=4)

    print("-" * 60)
    print(f"Message: {save_path}")


if __name__ == '__main__':
    main()