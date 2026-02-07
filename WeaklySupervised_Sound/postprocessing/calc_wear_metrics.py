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

# ================= 配置区域 =================
# 计算强监督预测结果的指标
# 1. 您提供的预测结果路径 (Unprocessed Results)
# 假设该目录下直接存放着 v_seg_loso_sbj_X.csv 文件
PRED_PATH = '/home/yinjiaxi/wstal/tal_for_har/logs/actionformer/2025-12-03 12:45:25_/'

# CONF_PATH = PRED_PATH下的cfg.txt
# 需要从cfg.txt中获取当前数据集的'dataset_name'和'init_rand_seed'

# 2. 您提供的标注文件路径 (Annotations)
ANNO_PATH = '/home/yinjiaxi/wstal/tal_for_har/data/wear/annotations/'

# 3. 原始传感器数据路径 (用于 convert_segments_to_samples)
# 根据您的目录结构推断，如果不对请修改
RAW_DATA_PATH = '/home/yinjiaxi/wstal/tal_for_har/data/wear/raw/inertial'

# Wear 数据集参数
DATASET = 'wear'
NUM_CLASSES = 19
SAMPLING_RATE = 50
INPUT_DIM = 12
# Wear 数据集通常有 18 个 subject (0-17)
SUBJECT_IDS = range(18)

# 想要测试的阈值列表
SCORE_THRES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


# ===========================================

def main():
    print(f"开始评估 Dataset: {DATASET}")
    print(f"预测路径: {PRED_PATH}")
    print(f"标注路径: {ANNO_PATH}")
    print("-" * 60)

    # 表头
    header = "{:<10} | {:<8} {:<8} {:<8} | {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} | {:<8}".format(
        "Threshold", "P", "R", "F1", "UR", "OR", "DR", "IR", "FR", "MR", "mAP"
    )
    print(header)
    print("-" * 110)

    for f in SCORE_THRES:
        # 初始化累加器
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

        # 遍历所有 Subject
        for i in SUBJECT_IDS:
            sbj_name = f'loso_sbj_{i}'
            json_file_path = os.path.join(ANNO_PATH, f'{sbj_name}.json')
            pred_file_path = os.path.join(PRED_PATH, f'v_seg_{sbj_name}.csv')

            # 检查文件是否存在
            if not os.path.exists(json_file_path):
                print(f"Warning: Annotation file not found: {json_file_path}")
                continue
            if not os.path.exists(pred_file_path):
                print(f"Warning: Prediction file not found: {pred_file_path}")
                continue

            valid_subject_count += 1

            # 1. 加载标注
            with open(json_file_path) as fi:
                file = json.load(fi)
            anno_db = file['database']
            labels = ['null'] + list(file['label_dict'])
            label_dict = dict(zip(labels, list(range(len(labels)))))

            # 2. 加载预测结果
            v_seg = pd.read_csv(pred_file_path, index_col=None, low_memory=False)

            # 3. 加载原始数据 (用于将时间段转换为帧级样本)
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

            # 4. 阈值过滤
            v_seg_filtered = v_seg[v_seg.score > f].copy()
            v_seg_filtered = v_seg_filtered.rename(
                columns={"video_id": "video-id", "t_start": "t-start", "t_end": "t-end"})

            # 5. 计算 mAP (Detection Metrics)
            det_eval = ANETdetection(json_file_path, 'validation', tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                                     verbose=False)
            v_mAP, _ = det_eval.evaluate(v_seg_filtered)
            all_mAP.append(v_mAP)

            # 6. 计算样本级指标 (Sample-wise Metrics & Misalignment)
            preds, gt, _ = convert_segments_to_samples(v_seg_filtered, v_data, SAMPLING_RATE, threshold=f)

            eval_labels = range(NUM_CLASSES)  # 包含 null 类

            # 计算 P, R, F1
            v_prec = precision_score(gt, preds, average=None, labels=eval_labels, zero_division=0)
            v_rec = recall_score(gt, preds, average=None, labels=eval_labels, zero_division=0)
            v_f1_score = f1_score(gt, preds, average=None, labels=eval_labels, zero_division=0)

            all_prec += v_prec
            all_recall += v_rec
            all_f1 += v_f1_score

            # 计算 Misalignment Ratios (UR, OR, DR, ...)
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

        # --- 汇总当前阈值的结果 (平均值) ---
        avg_mAP = np.mean(all_mAP) * 100

        # 分类指标平均 (通常不包含 null 类，或者看具体需求，这里取所有类平均)
        avg_prec = np.mean(all_prec) / valid_subject_count * 100
        avg_rec = np.mean(all_recall) / valid_subject_count * 100
        avg_f1 = np.mean(all_f1) / valid_subject_count * 100

        # 错位指标平均
        avg_ur = np.mean(all_ur) / valid_subject_count * 100
        avg_or = np.mean(all_or) / valid_subject_count * 100
        avg_dr = np.mean(all_dr) / valid_subject_count * 100
        avg_ir = np.mean(all_ir) / valid_subject_count * 100
        avg_fr = np.mean(all_fr) / valid_subject_count * 100
        avg_mr = np.mean(all_mr) / valid_subject_count * 100

        # 打印一行结果
        print(
            "{:<10} | {:<8.2f} {:<8.2f} {:<8.2f} | {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} | {:<8.2f}".format(
                f, avg_prec, avg_rec, avg_f1, avg_ur, avg_or, avg_dr, avg_ir, avg_fr, avg_mr, avg_mAP
            ))
#         现在的代码是直接打印输出结果，请改写代码，将计算的指标结果整理成json文件，写入文件夹/home/yinjiaxi/wstal/tal_for_har/result/，json的文件名就是数据集名称+seed（变量）


if __name__ == '__main__':
    main()