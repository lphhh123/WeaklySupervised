# -*- coding: utf-8 -*-
import torch
import os
import json
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTrain, WeaklySupervisedXRFV2DatasetTest
from models.CDur_model import CDur
from tool import ANETdetection

# ============================================================
# ============================================================
data_paths = [
    "/home/lipei/all_6_30_3/",
    "/home/lipei/all_5_30_3/",
    "/home/lipei/all_4_30_3/",
    "/home/lipei/all_2_30_3/"
]

config = {
    "path": {
        "checkpoint_root": "./checkpoints/xrfv2_cdur_loso",
        "result_root": "./result/xrfv2_cdur_loso",
        "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
        "dataset_root_path": "/home/lipei/WWADL/",
    },
    "training": {
        "num_classes": 30,
        "batch_size": 32,
        "epochs": 30,
        "lr": 1e-4,
    }
}


# ============================================================
# ============================================================
def get_segments_by_frames(probs, threshold=0.15, min_len=15):
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
                all_segments[c].append([s, e, float(score)])
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
def run_train_fold(train_paths, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    datasets = [WeaklySupervisedXRFV2DatasetTrain(p, config["path"]["mapping_path"], use_airpods=True) for p in
                train_paths]
    combined_train_ds = ConcatDataset(datasets)
    train_loader = DataLoader(combined_train_ds, batch_size=config["training"]["batch_size"], shuffle=True,num_workers=8,pin_memory=True)

    model = CDur(inputdim=36, outputdim=config["training"]["num_classes"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    print(f"\n开始训练 [样本总数: {len(combined_train_ds)}]")
    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}", leave=False)
        for x, _, y in pbar:

            x = x.transpose(1, 2)
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1:03d} 结束 | 平均 Loss: {avg_loss:.4f}")

    save_path = os.path.join(save_dir, "best_model.pth")
    torch.save(model.state_dict(), save_path)
    return save_path


# ============================================================
# ============================================================
@torch.no_grad()
def run_dual_test_fold(test_path, checkpoint_path, device, test_mode="test_full"):
    fold_config = config.copy()
    fold_config['path']['test_dataset_path'] = test_path

    model = CDur(inputdim=36, outputdim=config["training"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_ds = WeaklySupervisedXRFV2DatasetTest(config=fold_config, use_airpods=True)
    results_dict = {}

    for file_name, data_iter in tqdm(test_ds.dataset(), desc=f"推理中 ({test_mode})", leave=False):
        raw_predictions = [[] for _ in range(config["training"]["num_classes"])]
        video_annotations = []

        for clip_dict, seg_range in data_iter:
            offset = int(seg_range[0])
            x = clip_dict['imu'].to(device)
            if x.dim() == 2: x = x.unsqueeze(0)
            x = x.transpose(1, 2)  # [1, 2048, 36]

            _, frame_prob = model(x, upsample=True)
            frame_prob = frame_prob.squeeze(0).cpu().numpy()

            segments = get_segments_by_frames(frame_prob)
            for cls_idx, segs in enumerate(segments):
                label_name = test_ds.id_to_action.get(str(cls_idx), str(cls_idx))
                for (s, e, score) in segs:
                    g_s, g_e = int(s + offset), int(e + offset)
                    if test_mode == "test_full":
                        raw_predictions[cls_idx].append([g_s, g_e, score])
                    else:
                        video_annotations.append({"segment": [g_s, g_e], "label": label_name, "score": float(score)})

        if test_mode == "test_full":
            for cls_idx in range(config["training"]["num_classes"]):
                cls_segs = np.array(raw_predictions[cls_idx])
                if len(cls_segs) == 0: continue
                keep = soft_nms_functional(cls_segs)
                label_name = test_ds.id_to_action.get(str(cls_idx), str(cls_idx))
                for i in keep:
                    video_annotations.append(
                        {"segment": [int(cls_segs[i, 0]), int(cls_segs[i, 1])], "label": label_name,
                         "score": float(cls_segs[i, 2])})
        results_dict[file_name] = video_annotations
    return results_dict


# ============================================================
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics = {"test_window": {}, "test_full": {}}

    for i, test_path in enumerate(data_paths):
        sub_id = os.path.basename(os.path.normpath(test_path))
        train_paths = [p for j, p in enumerate(data_paths) if j != i]

        print(f"\n" + "=" * 70)
        print(f"FOLD {i + 1}/4 | 测试受试者: {sub_id}")
        print(f"=" * 70)


        fold_ckpt_dir = os.path.join(config["path"]["checkpoint_root"], sub_id)
        best_ckpt = run_train_fold(train_paths, fold_ckpt_dir, device)


        for mode in ["test_window", "test_full"]:
            res_dict = run_dual_test_fold(test_path, best_ckpt, device, test_mode=mode)


            mode_json = os.path.join(config["path"]["result_root"], f"{sub_id}_{mode}.json")
            os.makedirs(os.path.dirname(mode_json), exist_ok=True)
            with open(mode_json, 'w') as f:

                json.dump({
                    "version": "VERSION 1.3",
                    "results": res_dict,
                    "external_data": {}
                }, f)


            eval_gt = os.path.join(test_path, "imu_annotations.json")
            evaluator = ANETdetection(eval_gt, mode_json, subset="test", verbose=False, check_status=False)
            mAPs, avg_mAP, _ = evaluator.evaluate()

            all_metrics[mode][sub_id] = {"mAP": [round(float(m), 4) for m in mAPs], "avg": round(float(avg_mAP), 4)}
            print(f"--- [{mode}] 完成 | 平均 mAP: {avg_mAP:.4f}")


    with open(os.path.join(config["path"]["result_root"], "loso_final_metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print("\n>>> LOSO 实验结束。请查看 loso_final_metrics.json。")