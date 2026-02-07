# -*- coding: utf-8 -*-
import torch
import os
import json
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTrain, WeaklySupervisedXRFV2DatasetTest
from models.CDur_model import CDur
from tool import ANETdetection

torch.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# ============================================================
# ============================================================
class RobustBCELoss(torch.nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, clip_prob, labels):
        n_classes = clip_prob.shape[-1]
        with torch.no_grad():
            target = labels * (1 - self.label_smoothing) + (1 - labels) * self.label_smoothing / (n_classes - 1)

        loss = F.binary_cross_entropy(clip_prob, target)
        return loss


def frame_probs_to_segments(probs, fps, threshold=0.1, min_duration=0.2):
    """
    For weakly supervised tasks, threshold is usually set low (e.g., 0.1-0.2).
    """
    T, C = probs.shape
    segments = [[] for _ in range(C)]
    for c in range(C):
        binary = probs[:, c] > threshold
        diff = np.diff(np.concatenate(([0], binary.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            t_start = s / fps
            t_end = e / fps
            if (t_end - t_start) >= min_duration:
                score = np.mean(probs[s:e, c])
                segments[c].append([t_start, t_end, float(score)])
    return segments


# ============================================================
# ============================================================
def train_cdur_xrfv2(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_airpods = config["training"]["use_airpods"]
    in_channels = 36 if use_airpods else 30
    num_classes = config["training"]["num_classes"]

    model = CDur(inputdim=in_channels, outputdim=num_classes,
                 temppool=config["model"]["temppool"]).to(device)

    train_ds = WeaklySupervisedXRFV2DatasetTrain(
        dataset_dir=config["path"]["train_dataset_path"],
        mapping_path=config["path"]["mapping_path"],
        split="train",
        use_airpods=use_airpods
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=1e-5)
    criterion = RobustBCELoss(label_smoothing=0.05).to(device)

    os.makedirs(config["path"]["checkpoint_path"], exist_ok=True)
    best_loss = float('inf')

    print(f" XRFV2 Training Started | Input: {in_channels} | Classes: {num_classes}")

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        for batch_idx, (data, _, labels) in enumerate(pbar):
            data = data.transpose(1, 2).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            clip_prob, _ = model(data)

            loss = criterion(clip_prob, labels)

            if torch.isnan(loss):
                print("Warning: NaN Loss detected!")
                continue

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.7f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.7f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(config["path"]["checkpoint_path"], "best_model.pth"))

    return os.path.join(config["path"]["checkpoint_path"], "best_model.pth")


# ============================================================
# ============================================================
# ============================================================
# ============================================================

def soft_nms_functional(dets, sigma=0.5, thresh=0.001):
    """
    dets: [[start, end, score], ...]
    """
    if len(dets) == 0: return []

    tstart = dets[:, 0]
    tend = dets[:, 1]
    tscore = dets[:, 2]

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
        iou = inter / union

        weight = np.exp(-(iou * iou) / sigma)
        tscore[order[1:]] *= weight

        mask = tscore[order[1:]] > thresh
        order = order[1:][mask]
        if order.size > 0:
            new_order = tscore[order].argsort()[::-1]
            order = order[new_order]

    return res_indices


@torch.no_grad()
def test_cdur_xrfv2(config, checkpoint_path, test_mode="test_window"):
    """
    Supports test_window and optimized test_full (aggregated sliding predictions + NMS).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_airpods = config["training"].get("use_airpods", True)
    in_channels = 36 if use_airpods else 30
    num_classes = config["training"]["num_classes"]
    fps = 50

    model = CDur(inputdim=in_channels, outputdim=num_classes,
                 temppool=config["model"].get("temppool", "linear")).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_ds = WeaklySupervisedXRFV2DatasetTest(config=config, use_airpods=use_airpods)
    id2label = test_ds.id_to_action

    conf_thresh = config["testing"].get("conf_thresh", 0.01 if test_mode == "test_full" else 0.05)
    nms_sigma = config["testing"].get("nms_sigma", 0.3)

    results_cache = {}
    print(f"\nStarting test mode: {test_mode}")

    for file_path_raw, data_iter in tqdm(test_ds.dataset(), desc=f"Testing {test_mode}"):
        video_id = os.path.basename(file_path_raw)

        raw_predictions = [[] for _ in range(num_classes)]

        for clip_dict, seg_range in data_iter:
            x = clip_dict['imu'].T.unsqueeze(0).to(device)
            _, frame_prob = model(x, upsample=True)
            frame_prob = frame_prob.squeeze(0).cpu().numpy()

            segments = frame_probs_to_segments(frame_prob, fps, threshold=conf_thresh)
            offset_frames = seg_range[0]

            for cls_idx, segs in enumerate(segments):
                for (s_sec, e_sec, score) in segs:
                    start_frame = s_sec * fps + offset_frames
                    end_frame = e_sec * fps + offset_frames
                    raw_predictions[cls_idx].append([start_frame, end_frame, score])

        final_video_preds = []

        if test_mode == "test_full":
            for cls_idx in range(num_classes):
                cls_segs = np.array(raw_predictions[cls_idx])
                if len(cls_segs) == 0: continue

                keep_indices = soft_nms_functional(cls_segs, sigma=nms_sigma, thresh=conf_thresh)

                label_name = id2label.get(str(cls_idx), id2label.get(cls_idx, f"class_{cls_idx}"))
                for i in keep_indices:
                    final_video_preds.append({
                        "label": label_name,
                        "score": float(cls_segs[i, 2]),
                        "segment": [float(cls_segs[i, 0]), float(cls_segs[i, 1])]
                    })
        else:
            for cls_idx in range(num_classes):
                label_name = id2label.get(str(cls_idx), id2label.get(cls_idx, f"class_{cls_idx}"))
                for seg in raw_predictions[cls_idx]:
                    final_video_preds.append({
                        "label": label_name,
                        "score": float(seg[2]),
                        "segment": [float(seg[0]), float(seg[1])]
                    })

        results_cache[video_id] = final_video_preds

    output_data = {
        "version": "VERSION 1.3",
        "results": results_cache,
        "external_data": {}
    }

    res_name = f"prediction_{test_mode}.json"
    pred_path = os.path.join(config["path"]["result_path"], res_name)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)

    with open(pred_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_preds = sum(len(v) for v in results_cache.values())
    print(f"[{test_mode}] Total detections generated: {total_preds}")

    if total_preds == 0:
        print(f"Warning: no actions detected in {test_mode} mode.")
        return 0.0

    tious = [0.1, 0.2, 0.3, 0.4, 0.5]
    evaluator = ANETdetection(
        ground_truth_filename=test_ds.eval_gt,
        prediction_filename=pred_path,
        subset="test",
        tiou_thresholds=tious,
        verbose=False,
        check_status=False
    )

    mAPs, avg_mAP, _ = evaluator.evaluate()
    print(f"[{test_mode}] Average mAP: {avg_mAP:.4f}")
    return avg_mAP


# ============================================================
# ============================================================
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
            "use_airpods": True,
            "num_classes": 30,
            "batch_size": 16,
            "num_epochs": 80,
            "lr": 1e-3,
            "num_workers": 4
        },
        "testing": {
            "conf_thresh": 0.05,
        }
    }

    best_ckpt = train_cdur_xrfv2(config)

    test_cdur_xrfv2(config, best_ckpt, test_mode="test_full")
    test_cdur_xrfv2(config, best_ckpt, test_mode="test_window")
