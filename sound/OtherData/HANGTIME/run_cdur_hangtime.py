# OtherData/HANGTIME/run_cdur_hangtime.py
# -*- coding: utf-8 -*-
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
print("\n" + "="*30)
print(f"1. Is CUDA available? : {torch.cuda.is_available()}")
print(f"2. CUDA version       : {torch.version.cuda}")
print(f"3. Device Count       : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"4. Current Device     : {torch.cuda.current_device()}")
    print(f"5. Device Name        : {torch.cuda.get_device_name(0)}")
print("="*30 + "\n")
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

import contextlib
import io
import json
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tool import ANETdetection
from OtherData.HANGTIME.dataset_hangtime_ws import WeaklyHangtimeDataset
from OtherData.utils import _meta_get, set_seed, featbox_to_time_seconds, build_gt_for_anet, dump_config
from models.CDur_model import CDur, MilSEDCNN


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
# ============================================================
def train_cdur_one_fold_hangtime(config, fold: int, exp_name: str = "cdur_hangtime"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configs
    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 50))
    clip_sec = float(config.get("clip_sec", 3.0))
    in_channels = int(config.get("in_channels", 113))
    num_classes = int(config["num_classes"])
    eval_interval = config["training"].get("eval_interval", 2)

    # Model
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

    # Dataset
    loso_json = f"loso_sbj_{fold}.json"
    train_ds = WeaklyHangtimeDataset(
        dataset_dir=dataset_dir, loso_json=loso_json, mode="train",
        fps=fps, num_sensors=in_channels, clip_sec=clip_sec,
        clip_overlap=float(config.get("clip_overlap", 0.5)),
        num_classes=num_classes, normalize=True,
        stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
        neg_keep_ratio=float(config.get("neg_keep_ratio", 0.5)),
        return_meta=False, seed=int(config.get("seed", 2022)) + fold,
    )

    bs = int(config["training"].get("batch_size", 16))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=int(config["training"].get("num_workers", 4)),
                              pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = BCELossWithLabelSmoothing(label_smoothing=0.1).to(device)

    ckpt_dir = os.path.join(config["checkpoint_dir"], f"fold{fold}")
    os.makedirs(ckpt_dir, exist_ok=True)


    last_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_last.pth")
    best_map_path = os.path.join(ckpt_dir, f"{exp_name}_best_map.pth")

    best_avg_map = 0.0
    num_epochs = int(config["training"]["num_epochs"])

    print(f"\n[Train CDur] fold={fold} | Start Training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (sample_30s, labels) in enumerate(pbar):
            sample_30s = sample_30s.to(device)
            labels = labels.to(device).float()
            inputs = sample_30s.permute(0, 2, 1)  # [B, T, C]

            optimizer.zero_grad()
            clip_prob, frame_prob = model(inputs)
            loss = criterion(clip_prob, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * sample_30s.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_ds)
        lr = scheduler.get_last_lr()[0]
        print(f"[Fold {fold}] Epoch {epoch + 1} | avg_loss={avg_loss:.6f} | lr={lr:.6f}")
        scheduler.step()

        # Save Last Model
        torch.save(model.state_dict(), last_ckpt_path)

        # ====================================================

        # ====================================================
        if (epoch + 1) % eval_interval == 0:
            print(f"  >>> Evaluating at Epoch {epoch + 1}...")

            curr_avg_map, detail_maps = test_cdur_hangtime_multithresh(
                config, last_ckpt_path, fold, verbose=False
            )

            print(f"  >>> Epoch {epoch + 1} Result: Avg mAP (across thresholds) = {curr_avg_map:.4f}")

            if curr_avg_map > best_avg_map:
                best_avg_map = curr_avg_map
                torch.save(model.state_dict(), best_map_path)
                print(f"  ****** New Best mAP! Saved to {best_map_path} ******")

            model.train()

    print(f"\nTraining Finished. Best mAP: {best_avg_map:.4f}")
    return best_map_path


# ============================================================
# ============================================================
@torch.no_grad()
def test_cdur_hangtime_multithresh(config, checkpoint_path, fold: int, test_mode: str = "test_window",
                                   verbose: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 50))
    clip_sec = float(config.get("clip_sec", 3.0))
    in_channels = int(config.get("in_channels", 113))
    num_classes = int(config["num_classes"])

    if config.get("model_type", "CDur") == "CDur":
        model = CDur(inputdim=in_channels, outputdim=num_classes,
                     temppool=config["training"].get("pool_type", "linear"))
    else:
        model = MilSEDCNN(inputdim=in_channels, outputdim=num_classes,
                          temppool=config["training"].get("pool_type", "soft"))

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    # Load GT
    loso_json = f"loso_sbj_{fold}.json"
    ann_path = os.path.join(dataset_dir, "annotations", loso_json)
    with open(ann_path, "r") as f:
        js = json.load(f)
    id2label = {int(v): k for k, v in js.get("label_dict", {}).items()}

    ds = WeaklyHangtimeDataset(
        dataset_dir=dataset_dir, loso_json=loso_json, mode=test_mode,
        fps=fps, num_sensors=in_channels, clip_sec=clip_sec, clip_overlap=0.0,
        num_classes=num_classes, normalize=True,
        stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
        neg_keep_ratio=1.0, return_meta=True, seed=int(config.get("seed", 2022))
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)


    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    fold_dir = os.path.join(config["result_root"], f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)


    cached_predictions = []
    # if verbose: print(f"[Test] Inferencing on {len(ds)} samples...")

    for x, _, meta in loader:
        x = x.to(device)
        inputs = x.permute(0, 2, 1)
        clip_prob, frame_prob = model(inputs, upsample=True)
        frame_prob = frame_prob.squeeze(0).cpu().numpy()
        cached_predictions.append({
            "sbj": str(_meta_get(meta, "sbj")),
            "start_frame": int(_meta_get(meta, "start")),
            "probs": frame_prob
        })


    map_per_threshold = []


    gt_path = os.path.join(fold_dir, "gt_for_anet.json")
    build_gt_for_anet(ann_path, gt_path)
    tious = np.linspace(0.3, 0.7, 5)


    summary_logs = []

    for th in thresholds:
        results_cache = {}
        # Generate Proposals
        for item in cached_predictions:
            sbj = item["sbj"]
            if sbj not in results_cache: results_cache[sbj] = []
            segments_per_class = frame_probs_to_segments(item["probs"], fps, threshold=th)
            for cls_idx, segs in enumerate(segments_per_class):
                label_name = id2label.get(cls_idx, f"class_{cls_idx}")
                for (s_sec, e_sec, score) in segs:
                    abs_start = s_sec + (item["start_frame"] / fps)
                    abs_end = e_sec + (item["start_frame"] / fps)
                    results_cache[sbj].append({
                        "label": label_name, "score": float(score),
                        "segment": [float(abs_start), float(abs_end)]
                    })

        pred_path = os.path.join(fold_dir, f"pred_th{th:.1f}.json")
        with open(pred_path, "w") as f:
            json.dump({"version": "CDur", "results": results_cache, "external_data": {}}, f)


        if abs(th - 0.5) < 1e-5:

            custom_pred_name = f"predictions_{test_mode}.json"
            custom_pred_path = os.path.join(fold_dir, custom_pred_name)
            with open(custom_pred_path, "w") as f:
                json.dump({"version": "CDur", "results": results_cache, "external_data": {}}, f)
            if verbose:
                print(f"  > Saved specific prediction file: {custom_pred_name}")


        with contextlib.redirect_stdout(io.StringIO()):
            evaluator = ANETdetection(ground_truth_filename=gt_path, prediction_filename=pred_path,
                                      subset="test", tiou_thresholds=tious, verbose=False)
            mAPs, avg_mAP, _ = evaluator.evaluate()

        map_per_threshold.append(avg_mAP)
        summary_logs.append(f"tIoU={th:.2f} -> mAP={avg_mAP:.4f}")


    final_avg_map = np.mean(map_per_threshold)

    if verbose:
        print("-" * 50)
        print(f"Fold {fold} Evaluation Results:")
        for log in summary_logs:
            print(log)
        print(f"平均mAP: {final_avg_map:.4f}")
        print("=" * 50)

    return final_avg_map, map_per_threshold


# ============================================================
# ============================================================
def run_loso_cdur_hangtime(config):

    set_seed(int(config.get("seed", 2022)))
    os.makedirs(config["result_root"], exist_ok=True)
    dump_config(config, config["result_root"])

    folds = config.get("folds", list(range(24)))


    final_results_list = []


    json_save_path = os.path.join(config["result_root"], "final_results_summary.json")

    for fold in folds:
        print(f"\n{'=' * 20} Start Processing Fold {fold} {'=' * 20}")


        best_ckpt = train_cdur_one_fold_hangtime(config, fold)

        # ------------------------------------------------------

        # ------------------------------------------------------
        print(f"\n>>> [Fold {fold}] Running Default Test (test_window)...")
        avg_map_win, maps_win = test_cdur_hangtime_multithresh(
            config, best_ckpt, fold, test_mode="test_window", verbose=True
        )

        # ------------------------------------------------------

        # ------------------------------------------------------
        print(f"\n>>> [Fold {fold}] Running Full Test (test_full)...")
        avg_map_full, maps_full = test_cdur_hangtime_multithresh(
            config, best_ckpt, fold, test_mode="test_full", verbose=True
        )

        # ------------------------------------------------------

        # ------------------------------------------------------

        fold_record = {
            "fold": int(fold),

            "mAPs": [float(x) for x in maps_win],
            "avg_mAP": float(avg_map_win),


            "test_full": {
                "mAPs": [float(x) for x in maps_full],
                "avg_mAP": float(avg_map_full)
            }
        }

        final_results_list.append(fold_record)


        with open(json_save_path, "w") as f:
            json.dump(final_results_list, f, indent=2)

        print(f">>> Fold {fold} results saved to {json_save_path}")

    # ------------------------------------------------------

    # ------------------------------------------------------
    if len(final_results_list) > 0:
        avg_win = np.mean([item["avg_mAP"] for item in final_results_list])
        avg_full = np.mean([item["test_full"]["avg_mAP"] for item in final_results_list])

        print("\n" + "#" * 60)
        print(f"Overall Performance across {len(final_results_list)} folds:")
        print(f"  > Average mAP (Window): {avg_win:.4f}")
        print(f"  > Average mAP (Full)  : {avg_full:.4f}")
        print(f"Final JSON saved to: {json_save_path}")
        print("#" * 60)


if __name__ == "__main__":
    config = {
        "seed": 2022,
        "exp_name": "cdur_hangtime",
        "model_type": "CDur",

        "dataset_dir": "/home/lipei/TAL_data/hangtime/",
        "checkpoint_dir": "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/hangtime_cdur_10_5s_2022",
        "result_root": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/hangtime_cdur_10_5s_2022",

        "folds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],

        "fps": 50,
        "clip_sec": 3.0,
        "in_channels": 3,
        "num_classes": 5,
        "stats_dirname": "loso_norm_stats_json",

        "training": {
            "batch_size": 32,
            "num_epochs": 80,
            "lr": 1e-4,
            "pool_type": "linear",
            "num_workers": 4,
            "eval_interval": 5,
        },


        "testing": {}
    }

    run_loso_cdur_hangtime(config)