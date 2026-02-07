# -*- coding: utf-8 -*-
import torch
import os
import json
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.ndimage

from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTrain, WeaklySupervisedXRFV2DatasetTest
from models.CDur_model import CDur
from tool import ANETdetection

torch.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# ============================================================
# 1) 损失函数
# ============================================================
class RobustBCELoss(torch.nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, clip_prob, labels):
        target = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy(clip_prob, target)


# ============================================================
# 2) 后处理：高斯平滑 + 动作膨胀
# ============================================================
def frame_probs_to_segments_v2(probs, fps, threshold=0.1, min_duration=0.5, sigma=2.0):
    T, C = probs.shape
    segments = [[] for _ in range(C)]
    smoothed_probs = scipy.ndimage.gaussian_filter1d(probs, sigma=sigma, axis=0)

    for c in range(C):
        binary = smoothed_probs[:, c] > threshold
        if not np.any(binary): continue
        diff = np.diff(np.concatenate(([0], binary.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            t_start = s / fps
            t_end = e / fps
            if (t_end - t_start) >= min_duration:
                # 补偿 0.3s 响应延迟
                t_start = max(0, t_start - 0.3)
                t_end = t_end + 0.3
                score = np.max(smoothed_probs[s:e, c])
                segments[c].append([t_start, t_end, float(score)])
    return segments


def soft_nms_functional(dets, sigma=0.5, thresh=0.001):
    if len(dets) == 0: return []
    dets = np.array(dets)
    tstart, tend, tscore = dets[:, 0], dets[:, 1], dets[:, 2]
    order = tscore.argsort()[::-1]
    res_indices = []
    while order.size > 0:
        i = order[0]
        res_indices.append(i)
        if order.size == 1: break
        xx1 = np.maximum(tstart[i], tstart[order[1:]])
        xx2 = np.minimum(tend[i], tend[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        areas = tend - tstart
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)
        weight = np.exp(-(iou * iou) / sigma)
        tscore[order[1:]] *= weight
        mask = tscore[order[1:]] > thresh
        order = order[1:][mask]
        if order.size > 0:
            order = order[tscore[order].argsort()[::-1]]
    return res_indices


# ============================================================
# 3) 训练与测试
# ============================================================
def train_cdur_xrfv2(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CDur(inputdim=36 if config["training"]["use_airpods"] else 30,
                 outputdim=config["training"]["num_classes"],
                 temppool=config["model"]["temppool"]).to(device)

    train_ds = WeaklySupervisedXRFV2DatasetTrain(
        dataset_dir=config["path"]["train_dataset_path"],
        mapping_path=config["path"]["mapping_path"],
        split="train", use_airpods=config["training"]["use_airpods"]
    )
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=1e-5)
    criterion = RobustBCELoss(label_smoothing=0.05).to(device)

    best_loss = float('inf')
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for data, _, labels in pbar:
            data = data.transpose(1, 2).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            clip_prob, _ = model(data)
            loss = criterion(clip_prob, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(config["path"]["checkpoint_path"], "best_model.pth"))
    return os.path.join(config["path"]["checkpoint_path"], "best_model.pth")


@torch.no_grad()
def test_cdur_xrfv2_improved(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config["training"]["num_classes"]
    fps = 50

    model = CDur(inputdim=36 if config["training"]["use_airpods"] else 30, outputdim=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    test_ds = WeaklySupervisedXRFV2DatasetTest(config=config, use_airpods=config["training"]["use_airpods"])
    id2label = test_ds.id_to_action

    # 多尺度阈值，增加 Recall
    test_thresholds = [0.01, 0.03, 0.05, 0.1, 0.15]
    results_cache = {}

    for file_path_raw, data_iter in tqdm(test_ds.dataset(), desc="Inference"):
        # --- 重要：ID 必须包含 .h5 以匹配 GT 文件 ---
        video_id = os.path.basename(file_path_raw)
        if not video_id.endswith(".h5"):
            # 如果 Dataset 给的是 .npy，强行转成 GT 期待的 .h5
            video_id = os.path.splitext(video_id)[0] + ".h5"

        raw_predictions = [[] for _ in range(num_classes)]

        for clip_dict, seg_range in data_iter:
            x = clip_dict['imu'].T.unsqueeze(0).to(device)
            _, frame_prob = model(x, upsample=True)
            frame_prob = frame_prob.squeeze(0).cpu().numpy()

            for thr in test_thresholds:
                segments = frame_probs_to_segments_v2(frame_prob, fps, threshold=thr, sigma=2.0)
                offset_frames = seg_range[0]
                for cls_idx, segs in enumerate(segments):
                    for (s_sec, e_sec, score) in segs:
                        # 转换回全局帧数 (因为 GT 里是 0, 250 这样的帧编号)
                        # 如果评估器 ANET 内部转了秒，则这里保留秒。
                        # 根据你的 GT，segment 是 [0, 250]，这通常代表帧。
                        start_f = s_sec * fps + offset_frames
                        end_f = e_sec * fps + offset_frames
                        raw_predictions[cls_idx].append([start_f, end_f, score])

        final_video_preds = []
        for cls_idx in range(num_classes):
            cls_segs = np.array(raw_predictions[cls_idx])
            if len(cls_segs) == 0: continue

            keep_indices = soft_nms_functional(cls_segs, sigma=0.5, thresh=0.01)
            label_name = id2label.get(str(cls_idx), id2label.get(cls_idx, f"class_{cls_idx}"))
            for i in keep_indices:
                final_video_preds.append({
                    "label": label_name,
                    "score": float(cls_segs[i, 2]),
                    "segment": [float(cls_segs[i, 0]), float(cls_segs[i, 1])]
                })
        results_cache[video_id] = final_video_preds

    pred_path = os.path.join(config["path"]["result_path"], "prediction_final_v2.json")
    with open(pred_path, 'w') as f:
        json.dump({"version": "VERSION 1.3", "results": results_cache}, f, indent=2)

    # 运行评估
    evaluator = ANETdetection(
        ground_truth_filename=test_ds.eval_gt,
        prediction_filename=pred_path,
        subset="test",
        tiou_thresholds=[0.1, 0.3, 0.5],
        check_status=False
    )
    mAPs, avg_mAP, _ = evaluator.evaluate()
    print(f"\n>>> Final Improved mAP: {avg_mAP:.4f}")


if __name__ == "__main__":
    config = {
        "path": {
            "train_dataset_path": "/home/lipei/XRFV2/",
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/xrfv2_cdur",
            "result_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_cdur"
        },
        "model": {"temppool": "linear"},
        "training": {
            "use_airpods": True, "num_classes": 30, "batch_size": 16,
            "num_epochs": 80, "lr": 5e-4
        }
    }

    best_ckpt = train_cdur_xrfv2(config)
    test_cdur_xrfv2_improved(config, best_ckpt)