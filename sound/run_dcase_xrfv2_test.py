import os
import json
import torch
import numpy as np
from tqdm import tqdm
import copy

# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTest
from models.DCASE_CRNN_XRFV2 import CRNN


# ========================== 1. 时间轴非极大值抑制 (NMS) ==========================
def temporal_nms(detections, iou_threshold=0.3):
    if not detections: return []
    final_results = []
    labels = set([d['label'] for d in detections])
    for label in labels:
        label_dets = [d for d in detections if d['label'] == label]
        # 按得分从高到低排序
        label_dets = sorted(label_dets, key=lambda x: x['score'], reverse=True)
        keep = []
        while label_dets:
            curr = label_dets.pop(0)
            keep.append(curr)
            remaining = []
            for det in label_dets:
                s1, e1 = curr['segment']
                s2, e2 = det['segment']
                inter = max(0, min(e1, e2) - max(s1, s2))
                union = (e1 - s1) + (e2 - s2) - inter
                iou = inter / union if union > 0 else 0
                if iou < iou_threshold:
                    remaining.append(det)
            label_dets = remaining
        final_results.extend(keep)
    return final_results


# ========================== 2. 结构化后处理函数 ==========================
def post_process_to_dict(predictions, threshold, offset_frames, id_to_action, min_duration=10):
    n_class, n_frames = predictions.shape
    video_annotations = []

    for cls_idx in range(n_class):
        probs = predictions[cls_idx, :]
        mask = probs > threshold

        if not np.any(mask): continue

        diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        label_name = id_to_action.get(str(cls_idx), str(cls_idx))
        for s, e in zip(starts, ends):
            duration = e - s
            if duration < min_duration:
                continue

            abs_s = int(s + offset_frames)
            abs_e = int(e + offset_frames)
            score = float(probs[s:e].mean())

            video_annotations.append({
                "segment": [abs_s, abs_e],
                "label": label_name,
                "score": round(score, 4)
            })
    return video_annotations


# ========================== 3. 核心推理函数 ==========================
def run_pure_test(config, checkpoint_path, test_mode="full"):
    # --- 差异化参数设置 ---
    if test_mode == "full":
        # Full 模式：全局置信度可能被平摊，门槛稍微设低一点，去搜寻长动作
        current_threshold = 0.45
        current_iou = 0.1  # 2000帧几乎不重叠，IoU 设极低只合并微小重叠
        win_size = 2000
        hop_size = 2000
    else:
        # Window 模式：局部响应强，提高门槛过滤瞬时噪声，增加 IoU 允许 NMS 更好的合并
        current_threshold = 0.65
        current_iou = 0.4
        win_size = config["testing"]["window_size"]
        hop_size = config["testing"]["hop_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(n_in_channel=1, nclass=config["training"]["num_classes"], **config["model"]["params"]).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 找不到模型权重文件 {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 更新配置
    test_config = copy.deepcopy(config)
    test_config["testing"]["window_size"] = win_size
    test_config["testing"]["hop_size"] = hop_size

    test_ds = WeaklySupervisedXRFV2DatasetTest(config=test_config, use_airpods=config["training"]["use_airpods"])
    id_to_action = test_ds.id_to_action

    results = {}
    print(f"\n🔍 模式: {test_mode.upper()} | Threshold: {current_threshold} | NMS IoU: {current_iou}")

    with torch.no_grad():
        for file_name, data_iter in tqdm(test_ds.dataset(), desc=f"Mode: {test_mode}"):
            video_annotations = []
            for clip_dict, info in data_iter:
                data = clip_dict['imu'].unsqueeze(0).to(device)
                strong_out, _ = model(data)
                pred_probs = strong_out.squeeze(0).cpu().numpy()

                offset_frames = int(info[0]) if isinstance(info, (list, tuple, np.ndarray)) else 0

                clip_results = post_process_to_dict(
                    pred_probs,
                    threshold=current_threshold,
                    offset_frames=offset_frames,
                    id_to_action=id_to_action,
                    min_duration=10 if test_mode == "full" else 5
                )
                video_annotations.extend(clip_results)

            # 执行 NMS 合并
            video_annotations = temporal_nms(video_annotations, iou_threshold=current_iou)
            video_annotations.sort(key=lambda x: x["segment"][0])
            results[file_name] = video_annotations

    save_path = os.path.join(config["path"]["result_path"], f"predictions_test_{test_mode}.json")
    with open(save_path, 'w') as f:
        json.dump({"version": "VERSION 1.3", "results": results, "external_data": {}}, f, indent=4)
    print(f"✅ [{test_mode}] 结果保存至: {save_path}")


# ========================== 4. 主程序 ==========================
def main():
    base_config = {
        "path": {
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/xrfv2_dcase_2024",
            "result_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_2024"
        },
        "model": {
            "params": {
                "rnn_type": "BGRU", "n_RNN_cell": 128, "n_layers_RNN": 2,
                "dropout": 0.5, "attention": True, "specaugm_t_l": 10, "specaugm_f_l": 2
            }
        },
        "training": {"use_airpods": True, "num_classes": 30},
        "testing": {
            "window_size": 200,
            "hop_size": 50  # 减小步长，增加 window 模式的重叠度，使结果更精细
        }
    }

    os.makedirs(base_config["path"]["result_path"], exist_ok=True)
    # === 新增：CRNN 模型参数量统计 ===
    print("\n" + "=" * 50)
    print("📊 CRNN 模型结构与参数量统计:")
    from models.DCASE_CRNN_XRFV2 import CRNN

    # 根据 config 实例化模型
    model_stats = CRNN(
        n_in_channel=1,
        nclass=base_config["training"]["num_classes"],
        **base_config["model"]["params"]
    )

    # 1. 统计总数
    total_params = sum(p.numel() for p in model_stats.parameters())
    trainable_params = sum(p.numel() for p in model_stats.parameters() if p.requires_grad)

    # 2. 分模块统计 (CRNN 核心三部分)
    cnn_params = sum(p.numel() for n, p in model_stats.named_parameters() if "cnn" in n.lower())
    rnn_params = sum(p.numel() for n, p in model_stats.named_parameters() if "rnn" in n.lower() or "gru" in n.lower())
    att_params = sum(p.numel() for n, p in model_stats.named_parameters() if "att" in n.lower())

    print(f"🔹 总参数量 (Total): {total_params:,}")
    print(f"🔹 可训练参数 (Trainable): {trainable_params:,}")
    print(f"--------------------------------------------------")
    print(f"🔸 CNN 卷积层部分: {cnn_params:,} ({cnn_params / total_params:.1%})")
    print(f"🔸 RNN/GRU 循环层部分: {rnn_params:,} ({rnn_params / total_params:.1%})")
    print(f"🔸 Attention 机制部分: {att_params:,} ({att_params / total_params:.1%})")
    print(f"🔹 模型估算大小: {total_params * 4 / (1024 ** 2):.2f} MB")
    print("=" * 50 + "\n")
    # ===============================

    exp_name = "xrfv2_dcase_crnn_v1"
    best_ckpt_path = os.path.join(base_config["path"]["checkpoint_path"], f"{exp_name}_best.pth")

    # 同时执行两个模式
    run_pure_test(base_config, best_ckpt_path, test_mode="window")
    run_pure_test(base_config, best_ckpt_path, test_mode="full")


if __name__ == "__main__":
    main()