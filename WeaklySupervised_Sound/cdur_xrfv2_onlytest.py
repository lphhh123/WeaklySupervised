# -*- coding: utf-8 -*-
import torch
import os
import json
import numpy as np
from tqdm import tqdm

from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTest
from models.CDur_model import CDur
from tool import ANETdetection

torch.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# ============================================================
# ============================================================
def get_segments_by_frames(probs, threshold=0.15, min_len=10):
    T, C = probs.shape
    all_segments = [[] for _ in range(C)]
    mask = probs > threshold
    for c in range(C):
        m = mask[:, c]
        if not np.any(m): continue
        diff = np.diff(np.concatenate(([0], m.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) >= min_len:
                score = np.mean(probs[s:e, c])
                all_segments[c].append([int(s), int(e), float(score)])
    return all_segments


def soft_nms_functional(dets, sigma=0.5, thresh=0.001):
    if len(dets) == 0: return []
    dets = np.array(dets)
    tstart, tend, tscore = dets[:, 0], dets[:, 1], dets[:, 2].copy()
    order = tscore.argsort()[::-1]
    res_indices = []
    while order.size > 0:
        i = order[0]
        res_indices.append(i)
        if order.size == 1: break
        xx1 = np.maximum(tstart[i], tstart[order[1:]])
        xx2 = np.minimum(tend[i], tend[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        union = (tend[i] - tstart[i]) + (tend[order[1:]] - tstart[order[1:]]) - inter
        iou = inter / (union + 1e-6)
        tscore[order[1:]] *= np.exp(-(iou * iou) / sigma)
        mask = tscore[order[1:]] > thresh
        order = order[1:][mask]
        if order.size > 0:
            order = order[tscore[order].argsort()[::-1]]
    return res_indices


# ============================================================
# ============================================================
@torch.no_grad()
def test_dual_logic(config, checkpoint_path, test_mode="test_full"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config["training"]["num_classes"]

    ACT_THRESHOLD = 0.15
    MIN_FRAME_LEN = 15

    model = CDur(inputdim=36, outputdim=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_ds = WeaklySupervisedXRFV2DatasetTest(config=config, use_airpods=True)
    results_dict = {}

    for file_path_raw, data_iter in tqdm(test_ds.dataset(), desc=f"Mode: {test_mode}"):
        video_id = os.path.basename(file_path_raw)
        raw_predictions = [[] for _ in range(num_classes)]
        video_annotations = []

        for clip_dict, seg_range in data_iter:
            offset = int(seg_range[0])
            x = clip_dict['imu']
            if x.shape[0] in [30, 36]: x = x.T
            if x.shape[1] == 30:
                x = torch.cat([torch.tensor(x), torch.zeros((x.shape[0], 6))], dim=1)
            elif x.shape[1] > 36:
                x = x[:, :36]

            x = torch.tensor(x).float().unsqueeze(0)
            x = (x - x.mean()) / (x.std() + 1e-5)
            _, frame_prob = model(x.to(device), upsample=True)
            frame_prob = frame_prob.squeeze(0).cpu().numpy()

            segments = get_segments_by_frames(frame_prob, threshold=ACT_THRESHOLD, min_len=MIN_FRAME_LEN)

            for cls_idx, segs in enumerate(segments):
                label_name = test_ds.id_to_action.get(str(cls_idx), str(cls_idx))
                for (s, e, score) in segs:
                    g_s, g_e = int(s + offset), int(e + offset)
                    if test_mode == "test_full":
                        raw_predictions[cls_idx].append([g_s, g_e, score])
                    else:
                        video_annotations.append({"segment": [g_s, g_e], "label": label_name, "score": float(score)})

        if test_mode == "test_full":
            for cls_idx in range(num_classes):
                cls_segs = np.array(raw_predictions[cls_idx])
                if len(cls_segs) == 0: continue
                keep = soft_nms_functional(cls_segs, sigma=0.5, thresh=0.01)
                label_name = test_ds.id_to_action.get(str(cls_idx), str(cls_idx))
                for i in keep:
                    video_annotations.append({
                        "segment": [int(cls_segs[i, 0]), int(cls_segs[i, 1])],
                        "label": label_name,
                        "score": float(cls_segs[i, 2])
                    })
        results_dict[video_id] = video_annotations

    final_output = {
        "version": "VERSION 1.3",
        "results": results_dict,
        "external_data": {}
    }

    out_path = os.path.join(config["path"]["result_path"], f"prediction_{test_mode}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    evaluator = ANETdetection(test_ds.eval_gt, out_path, subset="test", tiou_thresholds=tious, verbose=False,
                              check_status=False)
    mAPs, avg_mAP, _ = evaluator.evaluate()
    return mAPs, avg_mAP


if __name__ == "__main__":
    config = {
        "path": {
            "train_dataset_path": "/home/lipei/XRFV2/",
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "./checkpoints/xrfv2_cdur",
            "result_path": "./result/xrfv2_cdur"
        },
        "training": {"num_classes": 30}
    }
    best_ckpt = os.path.join(config["path"]["checkpoint_path"], "best_model.pth")
    if os.path.exists(best_ckpt):
        metrics = {}
        for mode in ["test_window", "test_full"]:
            mAPs, avg = test_dual_logic(config, best_ckpt, test_mode=mode)
            metrics[mode] = {"mAP": [round(float(m), 4) for m in mAPs], "avg": round(float(avg), 4)}
        with open(os.path.join(config["path"]["result_path"], "metrics_report.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        print("\nMetrics updated. JSON format adjusted for tool.py.")
    else:
        print("FileNotFound.")
