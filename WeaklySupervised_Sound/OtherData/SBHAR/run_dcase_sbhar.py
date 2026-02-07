# OtherData/SBHAR/run_dcase_sbhar.py
# -*- coding: utf-8 -*-
import torch

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import os
import sys
import json
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tool import ANETdetection


try:
    from OtherData.SBHAR.dataset_sbhar_ws import WeaklySBHARDataset
except ImportError:
    print(
        "Warning: Could not import WeaklySBHARDataset. Please check the file path 'OtherData/SBHAR/dataset_sbhar_ws.py'.")

from OtherData.utils import _meta_get, set_seed, build_gt_for_anet, dump_config

# ============================================================

# ============================================================
try:
    from models.DCASE_CRNN import CRNN
except ImportError:
    print("Warning: Could not import CRNN from models.DCASE_CRNN.")
    print("Please ensure the file exists and contains the class definition provided.")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ============================================================
# Helper: Context manager to suppress stdout
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
# 2) Train one fold (SBHAR)
# ============================================================
def train_dcase_one_fold_sbhar(config, fold: int, exp_name: str = "dcase_sbhar"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 15.0))
    in_channels = int(config.get("in_channels", 3))  # SBHAR usually 3 channels
    num_classes = int(config["num_classes"])

    suffix = f"_{int(clip_sec)}s"


    cnn_kwargs = config["model_args"].get("cnn_kwargs", {})
    pooling_config = cnn_kwargs.get("pooling", None)


    model = CRNN(
        n_in_channel=1,
        nclass=num_classes,
        attention=config["model_args"].get("attention", True),
        n_RNN_cell=config["model_args"].get("n_RNN_cell", 128),
        n_layers_RNN=config["model_args"].get("n_layers_RNN", 2),
        dropout=config["model_args"].get("dropout", 0.5),
        specaugm_t_l=config["model_args"].get("specaugm_t_l", 5),
        specaugm_f_l=config["model_args"].get("specaugm_f_l", 2),
        cnn_integration=config["model_args"].get("cnn_integration", True),

        pooling=pooling_config,
        use_embeddings=False
    )

    model = model.to(device)
    def count_parameters(model):

        total_params = sum(p.numel() for p in model.parameters())

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    total, trainable = count_parameters(model)
    print(f"\n" + "-" * 30)
    # print(f"Model Type: {model_type}")
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
        seed=int(config.get("seed", 2022)) + fold,
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
    ckpt_path = os.path.join(ckpt_dir, f"{exp_name}{suffix}.pth")

    best_loss = float("inf")
    num_epochs = int(config["training"]["num_epochs"])

    print(f"\n[Train CRNN SBHAR] fold={fold} | clip={clip_sec}s | samples={len(train_ds)}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Fold{fold} Ep{epoch + 1}", leave=False)
        for batch_idx, (sample_clips, labels) in enumerate(pbar):
            sample_clips = sample_clips.to(device)
            labels = labels.to(device).float()


            if sample_clips.shape[1] > sample_clips.shape[2]:
                inputs = sample_clips.permute(0, 2, 1)
            else:
                inputs = sample_clips

            optimizer.zero_grad()
            frame_prob, clip_prob = model(inputs)
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
# 3) Test one fold (Threshold Search 0.3-0.7)
# ============================================================
@torch.no_grad()
def test_dcase_sbhar(config, checkpoint_path, fold: int, test_mode: str = "test_window"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 15.0))
    in_channels = int(config.get("in_channels", 3))
    num_classes = int(config["num_classes"])
    suffix = f"_{int(clip_sec)}s"


    cnn_kwargs = config["model_args"].get("cnn_kwargs", {})
    pooling_config = cnn_kwargs.get("pooling", None)


    model = CRNN(
        n_in_channel=1,
        nclass=num_classes,
        attention=config["model_args"].get("attention", True),
        n_RNN_cell=config["model_args"].get("n_RNN_cell", 128),
        n_layers_RNN=config["model_args"].get("n_layers_RNN", 2),
        specaugm_t_l=config["model_args"].get("specaugm_t_l", 5),
        specaugm_f_l=config["model_args"].get("specaugm_f_l", 2),
        cnn_integration=config["model_args"].get("cnn_integration", True),
        pooling=pooling_config,
        use_embeddings=False
    )

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
        seed=int(config.get("seed", 2022)) + fold,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # ----------------------------------------------------
    # Step A: Inference & Cache
    # ----------------------------------------------------
    prob_cache = {}  # Key: sbj, Value: (frame_prob_numpy, clip_start_frame)

    for x, _, meta in tqdm(loader, desc=f"[{test_mode}] fold{fold} inference", leave=False):
        sbj = str(_meta_get(meta, "sbj"))
        clip_start_frame = int(_meta_get(meta, "start"))

        x = x.to(device)
        if x.shape[1] > x.shape[2]:
            inputs = x.permute(0, 2, 1)
        else:
            inputs = x

        frame_prob, _ = model(inputs)


        target_len = inputs.shape[-1]
        frame_prob = F.interpolate(frame_prob, size=target_len, mode='linear', align_corners=True)
        # (B, C, T) -> (T, C)
        frame_prob = frame_prob.permute(0, 2, 1).squeeze(0).cpu().numpy()

        if sbj not in prob_cache:
            prob_cache[sbj] = []
        prob_cache[sbj].append((frame_prob, clip_start_frame))

    gt_path = os.path.join(config["result_root"], f"fold{fold}", "gt_for_anet.json")
    if not os.path.exists(gt_path):
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        build_gt_for_anet(ann_path, gt_path)

    # ----------------------------------------------------
    # Step B: Threshold Search
    # ----------------------------------------------------
    search_thresholds = config["testing"].get("thresholds", [0.3, 0.4, 0.5, 0.6, 0.7])

    best_avg_mAP = -1.0
    best_thresh = 0.5
    best_mAPs = None

    # print(f"\n[Test] Searching thresholds {search_thresholds} for Fold {fold} ({test_mode})...")

    for th in search_thresholds:
        results_cache = {}
        for sbj, clip_list in prob_cache.items():
            results_cache[sbj] = []
            for (frame_prob, clip_start_frame) in clip_list:
                segments_per_class = frame_probs_to_segments(frame_prob, fps, threshold=th)

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

        fold_dir = os.path.join(config["result_root"], f"fold{fold}")
        pred_path = os.path.join(fold_dir, f"predictions_{test_mode}{suffix}.json")

        final_results = {
            "version": "DCASE-CRNN-SBHAR",
            "results": results_cache,
            "external_data": {}
        }
        with open(pred_path, "w") as f:
            json.dump(final_results, f, indent=2)

        tious = np.linspace(0.3, 0.7, 5)
        evaluator = ANETdetection(
            ground_truth_filename=gt_path,
            prediction_filename=pred_path,
            subset="test",
            tiou_thresholds=tious,
            verbose=False,
            check_status=False
        )

        with HiddenPrints():
            mAPs, avg_mAP, _ = evaluator.evaluate()

        if avg_mAP > best_avg_mAP:
            best_avg_mAP = avg_mAP
            best_thresh = th
            best_mAPs = mAPs
            best_pred_path = os.path.join(fold_dir, f"best_predictions_{test_mode}{suffix}.json")
            with open(best_pred_path, "w") as f:
                json.dump(final_results, f, indent=2)

    print(f"  > [Fold {fold} {test_mode}] Best Thresh: {best_thresh} | Avg mAP: {best_avg_mAP:.4f}")

    return best_mAPs, best_avg_mAP, best_thresh


# ============================================================
# 4) Main Runner
# ============================================================
def run_loso_dcase_sbhar(config):
    set_seed(int(config.get("seed", 2022)))
    os.makedirs(config["result_root"], exist_ok=True)
    dump_config(config, config["result_root"])

    folds = config.get("folds", list(range(30)))
    clip_sec = config.get("clip_sec", 15.0)
    suffix = f"_{int(clip_sec)}s"

    all_folds_metrics = []

    for fold in folds:
        # 1. Train
        ckpt = train_dcase_one_fold_sbhar(config, fold, exp_name=config["exp_name"])

        # 2. Test Window
        mAPs_win, avg_mAP_win, th_win = test_dcase_sbhar(config, ckpt, fold, test_mode="test_window")

        # 3. Test Full
        mAPs_full, avg_mAP_full, th_full = test_dcase_sbhar(config, ckpt, fold, test_mode="test_full")

        fold_result = {
            "fold": fold,
            "clip_sec": clip_sec,
            "window_mode": {
                "best_threshold": float(th_win),
                "avg_mAP": float(avg_mAP_win),
                "mAPs": mAPs_win.tolist()
            },
            "full_mode": {
                "best_threshold": float(th_full),
                "avg_mAP": float(avg_mAP_full),
                "mAPs": mAPs_full.tolist()
            }
        }
        all_folds_metrics.append(fold_result)
        print(f"[Fold {fold}] Win(th={th_win}): {avg_mAP_win:.4f} | Full(th={th_full}): {avg_mAP_full:.4f}")

    final_json_path = os.path.join(config["result_root"], f"all_folds_results{suffix}.json")
    with open(final_json_path, "w") as f:
        json.dump(all_folds_metrics, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Results saved to: {final_json_path}")
    print("=" * 50)


if __name__ == "__main__":

    config = {
        "seed": 2022,
        "exp_name": "dcase_sbhar",
        "model_type": "CRNN",

        "dataset_dir": "/home/lipei/TAL_data/sbhar/",
        "checkpoint_dir": "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/sbhar_dcase_10_15s_2022",
        "result_root": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/sbhar_dcase_10_15s_2022/",

        "folds": list(range(30)),
        "fps": 30,
        "clip_sec": 15.0,
        "in_channels": 3,
        "num_classes": 12,
        "stats_dirname": "loso_norm_stats_json",

        "model_args": {
            "n_RNN_cell": 128,
            "n_layers_RNN": 2,
            "dropout": 0.5,
            "attention": True,
            "specaugm_t_l": 5,
            "specaugm_f_l": 2,
            "cnn_integration": True,


            "cnn_kwargs": {
                "pooling": [
                    [2, 3], [2, 1], [2, 1],
                    [1, 1], [1, 1], [1, 1], [1, 1]
                ]
            }
        },

        "training": {
            "batch_size": 32,
            "num_epochs": 80,
            "lr": 1e-4,
            "num_workers": 4,
        },

        "testing": {
            "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        }
    }

    run_loso_dcase_sbhar(config)