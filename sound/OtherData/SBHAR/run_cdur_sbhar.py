# OtherData/SBHAR/run_cdur_sbhar.py
# -*- coding: utf-8 -*-
import torch

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import os
import sys
import json
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tool import ANETdetection
                                                                      
                                
from OtherData.SBHAR.dataset_sbhar_ws import WeaklySBHARDataset
from OtherData.utils import _meta_get, set_seed, build_gt_for_anet, dump_config

            
from models.CDur_model import CDur, MilSEDCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# ============================================================
                                                      
# ============================================================
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ============================================================
# 0) Loss Function
# ============================================================
class BCELossWithLabelSmoothing(torch.nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, clip_prob, labels):
        n_classes = clip_prob.shape[-1]
        with torch.no_grad():
            tar = labels * (1 - self.label_smoothing) + (1 - labels) * self.label_smoothing / (n_classes - 1)
        return F.binary_cross_entropy(clip_prob, tar)


# ============================================================
# 1) Utility: Convert Frame Probs to Segments
# ============================================================
def frame_probs_to_segments(probs, fps, threshold=0.5, min_duration=0.1):
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
            duration = t_end - t_start

            if duration >= min_duration:
                score = np.mean(probs[s:e, c])
                segments[c].append([t_start, t_end, score])

    return segments


# ============================================================
# 2) Train one fold
# ============================================================
def train_cdur_one_fold_sbhar(config, fold: int, exp_name: str = "cdur_sbhar"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 15.0))
    in_channels = int(config.get("in_channels", 3))
    num_classes = int(config["num_classes"])

    model_type = config.get("model_type", "CDur")
    if model_type == "CDur":
        model = CDur(inputdim=in_channels, outputdim=num_classes,
                     temppool=config["training"].get("pool_type", "linear"))
    else:
        model = MilSEDCNN(inputdim=in_channels, outputdim=num_classes,
                          temppool=config["training"].get("pool_type", "soft"))

    model = model.to(device)

    def count_parameters(model):
                
        total_params = sum(p.numel() for p in model.parameters())
                            
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    total, trainable = count_parameters(model)
    print(f"\n" + "-" * 30)
    print(f"Model Type: {model_type}")
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print("-" * 30 + "\n")

    loso_json = f"loso_sbj_{fold}.json"
    train_ds = WeaklySBHARDataset(
        dataset_dir=dataset_dir,
        loso_json=loso_json,
        mode="train",
        fps=fps,
        num_sensors=in_channels,
        clip_sec=clip_sec,
        clip_overlap=float(config.get("clip_overlap", 0.5)),
        num_classes=num_classes,
        normalize=True,
        stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
        neg_keep_ratio=float(config.get("neg_keep_ratio", 0.5)),
        return_meta=False,
        seed=int(config.get("seed", 2026)) + fold,
    )

    bs = int(config["training"].get("batch_size", 32))
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = BCELossWithLabelSmoothing(label_smoothing=0.1).to(device)

    ckpt_dir = os.path.join(config["checkpoint_dir"], f"fold{fold}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{exp_name}.pth")

    best_loss = float("inf")
    num_epochs = int(config["training"]["num_epochs"])

    print(f"\n[Train CDur SBHAR] fold={fold} | samples={len(train_ds)}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Fold{fold} Ep{epoch + 1}", leave=False)
        for batch_idx, (sample_clips, labels) in enumerate(pbar):
            sample_clips = sample_clips.to(device)
            labels = labels.to(device).float()
            inputs = sample_clips.permute(0, 2, 1)

            optimizer.zero_grad()
            clip_prob, frame_prob = model(inputs)
            loss = criterion(clip_prob, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * sample_clips.size(0)

        avg_loss = epoch_loss / len(train_ds)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)

        scheduler.step()

    print(f"  >>> Fold {fold} Finished. Best Loss: {best_loss:.4f} -> {ckpt_path}")
    return ckpt_path


# ============================================================
# 3) Test one fold
# ============================================================
@torch.no_grad()
def test_cdur_sbhar(config, checkpoint_path, fold: int, test_mode: str = "test_window"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 15.0))
    in_channels = int(config.get("in_channels", 3))
    num_classes = int(config["num_classes"])

    # Load Model
    model_type = config.get("model_type", "CDur")
    if model_type == "CDur":
        model = CDur(inputdim=in_channels, outputdim=num_classes,
                     temppool=config["training"].get("pool_type", "linear"))
    else:
        model = MilSEDCNN(inputdim=in_channels, outputdim=num_classes,
                          temppool=config["training"].get("pool_type", "soft"))

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Labels
    loso_json = f"loso_sbj_{fold}.json"
    ann_path = os.path.join(dataset_dir, "annotations", loso_json)
    with open(ann_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    label_dict = js.get("label_dict", {})
    id2label = {int(v): k for k, v in label_dict.items()}

    # Dataset
    ds = WeaklySBHARDataset(
        dataset_dir=dataset_dir,
        loso_json=loso_json,
        mode=test_mode,
        fps=fps,
        num_sensors=in_channels,
        clip_sec=clip_sec,
        clip_overlap=0.0,
        num_classes=num_classes,
        normalize=True,
        stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
        neg_keep_ratio=1.0,
        return_meta=True,
        seed=int(config.get("seed", 2026)) + fold,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    results_cache = {}
                              
    threshold = float(config.get("testing", {}).get("threshold", 0.5))
    fold_dir = os.path.join(config["result_root"], f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Inference Loop
    for x, _, meta in tqdm(loader, desc=f"[{test_mode}] fold{fold}", leave=False):
        sbj = str(_meta_get(meta, "sbj"))
        clip_start_frame = int(_meta_get(meta, "start"))

        if sbj not in results_cache: results_cache[sbj] = []

        x = x.to(device)
        inputs = x.permute(0, 2, 1)

        clip_prob, frame_prob = model(inputs, upsample=True)
        frame_prob = frame_prob.squeeze(0).cpu().numpy()

        segments_per_class = frame_probs_to_segments(frame_prob, fps, threshold=threshold)

        for cls_idx, segs in enumerate(segments_per_class):
            label_name = id2label.get(cls_idx, f"class_{cls_idx}")
            for (start_sec, end_sec, score) in segs:
                abs_start = start_sec + (clip_start_frame / fps)
                abs_end = end_sec + (clip_start_frame / fps)
                results_cache[sbj].append({
                    "label": label_name,
                    "score": float(score),
                    "segment": [float(abs_start), float(abs_end)]
                })

    pred_path = os.path.join(fold_dir, f"predictions_{test_mode}.json")
    final_results = {"version": "CDur-SBHAR-v1", "results": results_cache, "external_data": {}}
    with open(pred_path, "w") as f:
        json.dump(final_results, f, indent=2)

    gt_path = os.path.join(fold_dir, "gt_for_anet.json")
    build_gt_for_anet(ann_path, gt_path)

    # Thresholds: 0.3 to 0.7
    tious = np.linspace(0.3, 0.7, 5)

    evaluator = ANETdetection(
        ground_truth_filename=gt_path,
        prediction_filename=pred_path,
        subset="test",
        tiou_thresholds=tious,
        verbose=False,
        check_status=False
    )

    # Suppress output
    with HiddenPrints():
        mAPs, avg_mAP, _ = evaluator.evaluate()

    return mAPs, avg_mAP


# ============================================================
# 4) Main Runner
# ============================================================
def run_loso_cdur_sbhar(config):
    set_seed(int(config.get("seed", 2026)))
    os.makedirs(config["result_root"], exist_ok=True)
    dump_config(config, config["result_root"])

    folds = config.get("folds", list(range(30)))

                       
    all_folds_metrics = []

    for fold in folds:
        # 1. Train
        ckpt = train_cdur_one_fold_sbhar(config, fold)

                                      
        mAPs_win, avg_mAP_win = test_cdur_sbhar(config, ckpt, fold, test_mode="test_window")

                                                  
        mAPs_full, avg_mAP_full = test_cdur_sbhar(config, ckpt, fold, test_mode="test_full")

                            
        fold_result = {
            "fold": fold,
            "mAPs": mAPs_win.tolist(),  # numpy -> list
            "avg_mAP": float(avg_mAP_win),
            "test_full": {
                "mAPs": mAPs_full.tolist(),  # numpy -> list
                "avg_mAP": float(avg_mAP_full)
            }
        }

        all_folds_metrics.append(fold_result)

        print(f"[Fold {fold}] Win Avg: {avg_mAP_win:.4f} | Full Avg: {avg_mAP_full:.4f}")

    # ============================================================
                
    # ============================================================
    final_json_path = os.path.join(config["result_root"], "all_folds_results.json")
    with open(final_json_path, "w") as f:
        json.dump(all_folds_metrics, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Results saved to: {final_json_path}")
    print("=" * 50)


if __name__ == "__main__":
    config = {
        "seed": 2026,
        "exp_name": "cdur_sbhar",
        "model_type": "CDur",

                    
        "dataset_dir": "/home/lipei/TAL_data/sbhar/",
        "checkpoint_dir": "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/sbhar_cdur_10_15s_2026",
        "result_root": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/sbhar_cdur_10_15s_2026",

        "folds": list(range(30)),  # 30-fold CV

        "fps": 30,
        "clip_sec": 15.0,
        "in_channels": 3,
        "num_classes": 12,
        "stats_dirname": "loso_norm_stats_json",

        "training": {
            "batch_size": 32,
            "num_epochs": 80,
            "lr": 1e-4,
            "pool_type": "linear",
            "num_workers": 4,
        },
        "testing": {
            "threshold": 0.5,
        }
    }

    run_loso_cdur_sbhar(config)