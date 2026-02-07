# OtherData/Opportunity/run_dcase_opportunity.py
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

import json
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.DCASE_RNN import BidirectionalGRU

from tool import ANETdetection
from OtherData.Opportunity.dataset_opportunity_ws import WeaklyOpportunityDataset
from OtherData.utils import _meta_get, set_seed, build_gt_for_anet, dump_config


from models.DCASE_CRNN import CRNN




# ============================================================
# 0) Loss Function & Adapter
# ============================================================
class AdapterCRNN(CRNN):
    def __init__(self, n_sensors=113, **kwargs):

        super().__init__(**kwargs)


        with torch.no_grad():

            dummy_input = torch.zeros(1, self.n_in_channel, 10, n_sensors)


            x = self.cnn(dummy_input)
            _, out_ch, _, out_freq = x.shape


            real_rnn_input_dim = out_ch * out_freq


        if real_rnn_input_dim != self.cnn.nb_filters[-1]:
            print(f"[AdapterCRNN] Auto-fixing RNN dimension: {self.cnn.nb_filters[-1]} -> {real_rnn_input_dim}")
            original_rnn = self.rnn
            self.rnn = BidirectionalGRU(
                n_in=real_rnn_input_dim,
                n_hidden=original_rnn.rnn.hidden_size,
                dropout=kwargs.get('dropout_recurrent', 0),
                num_layers=original_rnn.rnn.num_layers
            )


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
    """
    probs: [Time, Classes] numpy array
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
            duration = t_end - t_start

            if duration >= min_duration:
                score = np.mean(probs[s:e, c])
                segments[c].append([t_start, t_end, score])
    return segments


# ============================================================
# 2) Train one fold
# ============================================================
def train_dcase_one_fold_opportunity(config, fold: int, exp_name: str = "dcase_opportunity"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 5.0))
    in_channels = int(config.get("in_channels", 113))
    num_classes = int(config["num_classes"])

    # ---- Model Setup (CRNN) ----
    train_cfg = config["training"]
    model = AdapterCRNN(
        n_sensors=in_channels,
        n_in_channel=1,
        nclass=num_classes,
        attention=True,
        activation="glu",
        dropout=float(train_cfg.get("dropout", 0.5)),
        train_cnn=True,
        rnn_type="BGRU",
        n_RNN_cell=int(train_cfg.get("rnn_hidden", 128)),
        n_layers_RNN=int(train_cfg.get("rnn_layers", 2)),
        dropout_recurrent=0,
        cnn_integration=False,
        specaugm_t_l=int(train_cfg.get("specaugm_t_l", 5)),
        specaugm_f_l=int(train_cfg.get("specaugm_f_l", 2))
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

    # ---- Dataset ----
    loso_json = f"loso_sbj_{fold}.json"
    train_ds = WeaklyOpportunityDataset(
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

    bs = int(config["training"].get("batch_size", 16))
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

    print("\n" + "=" * 80)
    print(f"[Train DCASE] fold={fold} | model=CRNN | device={device}")
    print(f"  train samples: {len(train_ds)} | batch={bs}")
    print("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (sample_30s, labels) in enumerate(pbar):
            sample_30s = sample_30s.to(device)
            labels = labels.to(device).float()

            inputs = sample_30s  # [Batch, 113, Time]

            optimizer.zero_grad()
            frame_prob, clip_prob = model(inputs)

            loss = criterion(clip_prob, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * sample_30s.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_ds)
        lr = scheduler.get_last_lr()[0]
        print(f"[Fold {fold}] Epoch {epoch + 1} | avg_loss={avg_loss:.6f} | lr={lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >>> saved best model -> {ckpt_path}")

        scheduler.step()

    return ckpt_path


# ============================================================
# 3) Test one fold (Universal: handles both Window and Full)
# ============================================================
# ============================================================
# 3) Test one fold (Universal: handles both Window and Full)
# ============================================================
@torch.no_grad()
def test_dcase_opportunity(config, checkpoint_path, fold: int, test_mode: str = "test_window"):
    """
    test_mode: "test_window" (sliding windows) or "test_full" (full sequence)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 5.0))
    in_channels = int(config.get("in_channels", 113))
    num_classes = int(config["num_classes"])

    # Load Model
    train_cfg = config["training"]
    model = AdapterCRNN(
        n_sensors=in_channels,
        n_in_channel=1,
        nclass=num_classes,
        attention=True,
        activation="glu",
        dropout=float(train_cfg.get("dropout", 0.5)),
        train_cnn=True,
        rnn_type="BGRU",
        n_RNN_cell=int(train_cfg.get("rnn_hidden", 128)),
        n_layers_RNN=int(train_cfg.get("rnn_layers", 2)),
        cnn_integration=False
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Label Map
    loso_json = f"loso_sbj_{fold}.json"
    ann_path = os.path.join(dataset_dir, "annotations", loso_json)
    with open(ann_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    label_dict = js.get("label_dict", {})
    id2label = {int(v): k for k, v in label_dict.items()}

    # Dataset
    ds = WeaklyOpportunityDataset(
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

    bs = 1
    loader = DataLoader(ds, batch_size=bs, shuffle=False)

    fold_dir = os.path.join(config["result_root"], f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    inference_buffer = []
    print(f"\n[Test DCASE] fold={fold} | mode={test_mode} | Running Inference...")



    MAX_INFERENCE_LEN = 1800

    for x, _, meta in tqdm(loader, desc=f"Inference ({test_mode})"):
        # x shape: [1, 113, Time]
        total_len = x.shape[-1]


        full_prob_list = []


        for start_t in range(0, total_len, MAX_INFERENCE_LEN):
            end_t = min(start_t + MAX_INFERENCE_LEN, total_len)


            chunk_x = x[..., start_t:end_t].to(device)


            frame_out, _ = model(chunk_x)


            frame_out = F.interpolate(frame_out, size=chunk_x.shape[-1], mode='linear', align_corners=False)


            # Squeeze batch dim -> permute -> cpu
            chunk_prob = frame_out.squeeze(0).permute(1, 0).cpu().numpy()

            full_prob_list.append(chunk_prob)


        full_prob = np.concatenate(full_prob_list, axis=0)

        sbj = str(_meta_get(meta, "sbj"))
        clip_start_frame = int(_meta_get(meta, "start"))

        inference_buffer.append({
            "sbj": sbj,
            "start_frame": clip_start_frame,
            "probs": full_prob
        })

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_maps = []

    gt_path = os.path.join(fold_dir, "gt_for_anet.json")
    if not os.path.exists(gt_path):
        build_gt_for_anet(ann_path, gt_path)

    tious = np.linspace(0.3, 0.7, 5)

    print(f"[Test DCASE] ({test_mode}) Calculating mAP for thresholds {test_thresholds}...")

    for th in test_thresholds:
        results_cache = {}
        for item in inference_buffer:
            sbj = item["sbj"]
            start_f = item["start_frame"]
            probs = item["probs"]

            if sbj not in results_cache:
                results_cache[sbj] = []

            segments_per_class = frame_probs_to_segments(probs, fps, threshold=th)

            for cls_idx, segs in enumerate(segments_per_class):
                label_name = id2label.get(cls_idx, f"class_{cls_idx}")
                for (start_sec, end_sec, score) in segs:
                    abs_start = start_sec + (start_f / fps)
                    abs_end = end_sec + (start_f / fps)

                    results_cache[sbj].append({
                        "label": label_name,
                        "score": float(score),
                        "segment": [float(abs_start), float(abs_end)]
                    })

        pred_path = os.path.join(fold_dir, f"predictions_{test_mode}_th{th:.1f}.json")
        final_results = {"version": f"DCASE-Opp-{test_mode}", "results": results_cache, "external_data": {}}
        with open(pred_path, "w") as f:
            json.dump(final_results, f)

        evaluator = ANETdetection(
            ground_truth_filename=gt_path,
            prediction_filename=pred_path,
            subset="test",
            tiou_thresholds=tious,
            verbose=False
        )
        _, avg_mAP, _ = evaluator.evaluate()

        print(f"  > Fold{fold} | {test_mode} | Th {th:.1f} : mAP = {avg_mAP:.4f}")
        threshold_maps.append(float(avg_mAP))

    final_avg_mAP = np.mean(threshold_maps)
    return threshold_maps, float(final_avg_mAP)


# ============================================================
# 4) Main Runner
# ============================================================
def run_loso_dcase_opportunity(config):
    set_seed(int(config.get("seed", 2026)))
    os.makedirs(config["result_root"], exist_ok=True)
    dump_config(config, config["result_root"])

    folds = config.get("folds", [0, 1, 2, 3])

    all_reports = []


    window_results_list = []
    full_results_list = []

    for fold in folds:

        ckpt = train_dcase_one_fold_opportunity(config, fold)


        maps_window, avg_window = test_dcase_opportunity(config, ckpt, fold, test_mode="test_window")
        window_results_list.append(maps_window)


        maps_full, avg_full = test_dcase_opportunity(config, ckpt, fold, test_mode="test_full")
        full_results_list.append(maps_full)

        report_entry = {
            "fold": int(fold),
            "test_window": {
                "mAPs": maps_window,
                "avg_mAP": avg_window
            },
            "test_full": {
                "mAPs": maps_full,
                "avg_mAP": avg_full
            }
        }
        all_reports.append(report_entry)


    report_path = os.path.join(config["result_root"], "loso_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Final Report -> {report_path}")


    window_results_np = np.array(window_results_list)
    avg_maps_window = np.mean(window_results_np, axis=0)
    final_loso_window = np.mean(avg_maps_window)

    full_results_np = np.array(full_results_list)
    avg_maps_full = np.mean(full_results_np, axis=0)
    final_loso_full = np.mean(avg_maps_full)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print("\n" + "=" * 60)
    print("FINAL LOSO RESULTS (Average over all folds)")
    print("=" * 60)

    print(">>> Test Mode: WINDOW (Sliding Clips)")
    for i, th in enumerate(thresholds):
        print(f"  Threshold {th:.1f} : mAP = {avg_maps_window[i]:.4f}")
    print(f"  Average mAP (0.3-0.7) : {final_loso_window:.4f}")

    print("-" * 60)

    print(">>> Test Mode: FULL (Untrimmed Sequence)")
    for i, th in enumerate(thresholds):
        print(f"  Threshold {th:.1f} : mAP = {avg_maps_full[i]:.4f}")
    print(f"  Average mAP (0.3-0.7) : {final_loso_full:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    config = {
        "seed": 2026,
        "exp_name": "dcase_opportunity",
        "model_type": "CRNN",

        "dataset_dir": "/home/lipei/TAL_data/opportunity/",
        "checkpoint_dir": "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/opportunity_dcase_10_3s_2026",
        "result_root": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/opportunity_dcase_10_3s_2026",

        "folds": [0, 1, 2, 3],

        "fps": 30,
        "clip_sec": 3.0,
        "in_channels": 113,
        "num_classes": 17,
        "stats_dirname": "loso_norm_stats_json",

        "training": {
            "batch_size": 32,
            "num_epochs": 80,
            "lr": 1e-4,
            "num_workers": 4,


            "dropout": 0.5,
            "rnn_hidden": 128,
            "rnn_layers": 2,


            "specaugm_t_l": 10,
            "specaugm_f_l": 2,
        },
        "testing": {}
    }

    run_loso_dcase_opportunity(config)