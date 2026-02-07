import os
import json
import torch
import numpy as np
from tqdm import tqdm

# 导入工具类
from tool import ANETdetection, load_label_mapping
from models.DCASE_CRNN_XRFV2 import CRNN
from dataset.dataset_XRFV2_loso import WeaklySupervisedXRFV2DatasetTest


# ========================== 1. 后处理与 NMS ==========================

def temporal_nms(predictions, thresh=0.4):
    """ 时序 NMS：合并跨 window 的重复预测 """
    if len(predictions) == 0: return []
    df = sorted(predictions, key=lambda x: x['score'], reverse=True)
    keep = []
    while len(df) > 0:
        m = df.pop(0)
        keep.append(m)
        remaining = []
        for n in df:
            if m['label'] == n['label']:
                start = max(m['segment'][0], n['segment'][0])
                end = min(m['segment'][1], n['segment'][1])
                intersection = max(0, end - start)
                union = (m['segment'][1] - m['segment'][0]) + (n['segment'][1] - n['segment'][0]) - intersection
                iou = intersection / union if union > 0 else 0
                if iou < thresh: remaining.append(n)
            else:
                remaining.append(n)
        df = remaining
    return keep


def post_process_for_anet(probs, id_to_label, threshold=0.3, offset_frame=0):
    """ 将模型输出转化为 ANET 字典格式，单位为帧 """
    n_class, n_frames = probs.shape
    results = []
    for cls_idx in range(n_class):
        label_name = id_to_label.get(cls_idx, "unknown")
        mask = probs[cls_idx, :] > threshold
        start_f = None
        for i in range(n_frames):
            if mask[i] and start_f is None:
                start_f = i
            elif not mask[i] and start_f is not None:
                results.append({
                    "segment": [float(start_f + offset_frame), float(i + offset_frame)],
                    "label": label_name,
                    "score": float(probs[cls_idx, start_f:i].max())
                })
                start_f = None
        if start_f is not None:
            results.append({
                "segment": [float(start_f + offset_frame), float(n_frames + offset_frame)],
                "label": label_name,
                "score": float(probs[cls_idx, start_f:].max())
            })
    return results


# ========================== 2. 核心评估函数 ==========================

def run_subject_evaluation(config, ckpt_path, mode="window"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载标签映射
    _, _, id_to_label = load_label_mapping(config["path"]["mapping_path"])

    # 模型初始化
    model = CRNN(n_in_channel=1, nclass=config["training"]["num_classes"], **config["model"]["params"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_ds = WeaklySupervisedXRFV2DatasetTest(config=config, use_airpods=config["training"]["use_airpods"])

    prediction_output = {"results": {}, "version": "VERSION 1.3", "external_data": {}}

    print(f"--- 正在执行推理 [{mode}] ---")
    with torch.no_grad():
        for file_name, data_iter in tqdm(test_ds.dataset(), desc=f"Infer {mode}"):
            video_raw_segs = []
            for clip_dict, _ in data_iter:
                data = clip_dict['imu'].unsqueeze(0).to(device)
                offset = clip_dict.get('offset', 0)  # 确保 Dataset 返回了起始帧偏移

                strong_out, _ = model(data)
                # (1, T, C) -> (C, T) + Sigmoid
                probs = torch.sigmoid(strong_out).squeeze(0).cpu().numpy().T

                segs = post_process_for_anet(probs, id_to_label, threshold=0.25, offset_frame=offset)
                video_raw_segs.extend(segs)

            # 聚合当前视频结果并做 NMS
            prediction_output["results"][file_name] = temporal_nms(video_raw_segs, thresh=0.4)

    # 保存预测 JSON
    pred_path = os.path.join(config["path"]["result_path"], f"pred_{mode}.json")
    with open(pred_path, 'w') as f:
        json.dump(prediction_output, f, indent=4)

    # 计算 mAP
    tiou_thresholds = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    # 假设你的 GT 文件包含了所有 Subject 的标注，或者在各 Subject 路径下有独立的 GT
    gt_path = config["path"]["gt_path"]

    evaluator = ANETdetection(
        ground_truth_filename=gt_path,
        prediction_filename=pred_path,
        tiou_thresholds=tiou_thresholds,
        subset='test',
        verbose=False
    )
    mAP_list, avg_mAP, ap_matrix = evaluator.evaluate()

    # 整理结果
    report = {
        "average_mAP": float(avg_mAP),
        "mAP_at_thresholds": {f"mAP@{t}": float(m) for t, m in zip(tiou_thresholds, mAP_list)}
    }

    report_path = os.path.join(config["path"]["result_path"], f"map_report_{mode}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    return avg_mAP, mAP_list


# ========================== 3. 主循环适配 ==========================

def run_loso_test():
    loso_folders = ["all_6_30_3", "all_5_30_3", "all_4_30_3", "all_2_30_3"]
    data_root = "/home/lipei/"
    # 这里需要指定你之前提到的全局 GT 或 Subject 独立的 GT
    # 如果每个文件夹下有自己的 gt，请在循环内修改
    global_gt_path = "/home/lipei/XRFV2/imu_annotations.json"

    summary_results = {}

    for folder in loso_folders:
        print(f"\n" + "★" * 20 + f" 测试 LOSO 实验: {folder} " + "★" * 20)
        curr_path = os.path.join(data_root, folder)

        config = {
            "path": {
                "test_dataset_path": curr_path,
                "dataset_root_path": "/home/lipei/WWADL/",
                "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
                "checkpoint_path": os.path.join(
                    "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/xrfv2_dcase_loso_2022", folder),
                "result_path": os.path.join("/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_loso_2022",
                                            folder),
                "gt_path": global_gt_path
            },
            "model": {"params": {"rnn_type": "BGRU", "n_RNN_cell": 128, "n_layers_RNN": 2, "dropout": 0.5,
                                 "attention": True}},
            "training": {"use_airpods": True, "num_classes": 30}
        }

        best_ckpt = os.path.join(config["path"]["checkpoint_path"], f"dcase_{folder}_best.pth")

        if not os.path.exists(best_ckpt):
            print(f"跳过 {folder}，未找到权重文件: {best_ckpt}")
            continue

        # 执行 Window 模式
        avg_w, list_w = run_subject_evaluation(config, best_ckpt, mode="window")
        # 执行 Full 模式
        avg_f, list_f = run_subject_evaluation(config, best_ckpt, mode="full")

        # 控制台打印
        print(f"\n[结果汇总 - {folder}]")
        print(f"Window Mode -> Avg mAP: {avg_w:.4f} | mAP@0.5: {list_w[2]:.4f}")
        print(f"Full Mode   -> Avg mAP: {avg_f:.4f} | mAP@0.5: {list_f[2]:.4f}")

        summary_results[folder] = {"window": avg_w, "full": avg_f}

    # 最终汇总
    print("\n" + "=" * 60)
    print("LOSO 跨人实验最终 mAP 统计")
    print(f"{'Folder':<15} | {'Window mAP':<12} | {'Full mAP':<12}")
    for f, v in summary_results.items():
        print(f"{f:<15} | {v['window']:<12.4f} | {v['full']:<12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_loso_test()