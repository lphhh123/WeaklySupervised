# Opportunity/run_pcl_hangtime.py
# -*- coding: utf-8 -*-
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import os
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from tool import softnms_v2, ANETdetection
from OtherData.HANGTIME.dataset_hangtime_ws import WeaklyHangtimeDataset
from pre_train.pre_model import CNN1DBackbone
from models.PCL_OICR_model import IMU_PCL_OICR
from OtherData.utils import _meta_get, set_seed, featbox_to_time_seconds, build_gt_for_anet, ProposalWrappedDataset, \
    GlobalBackboneWrapper, generate_proposal_boxes

# ============================================================
# train one fold
# ============================================================
def train_pcl_oicr_one_fold_hangtime(config, fold: int, exp_name: str = "pcl_oicr_opportunity"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 30.0))
    in_channels = int(config.get("in_channels", 113))
    num_classes = int(config["num_classes"])

    # ---- load 3s backbone (fold-specific pretrain) ----
    backbone = CNN1DBackbone(in_channels=in_channels, feat_dim=512).to(device)
    pretrain_path = os.path.join(
        config["pretrained_dir"],
        f"hangtime_{config.get('pretrained_model_name','CNN1D')}_pretrained_loso_sbj_{fold}.pth"
    )
    if not os.path.isfile(pretrain_path):
        raise FileNotFoundError(f"pretrain_path not found: {pretrain_path}")
    backbone.load_state_dict(torch.load(pretrain_path, map_location=device))
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # ---- wrapper backbone ----
    win_len = int(config.get("seg_win_len", 90))
    seg_stride = int(config.get("seg_stride", 45))
    pretrained_backbone = GlobalBackboneWrapper(
        backbone, win_len=win_len, seg_stride=seg_stride, chunk=int(config.get("wrapper_chunk", 256))
    ).to(device)
    pretrained_backbone.eval()

    # ---- PCL/OICR model ----
    model = IMU_PCL_OICR(
        feat_dim=512,
        num_classes=num_classes,
        refine_times=int(config["training"].get("refine_times", 3)),
        use_pcl=bool(config["training"].get("use_pcl", False)),
        fg_thresh=float(config["training"].get("fg_thresh", 0.5)),
        bg_thresh=float(config["training"].get("bg_thresh", 0.1)),
        graph_iou_thresh=float(config["training"].get("graph_iou_thresh", 0.5)),
        max_pc_num=int(config["training"].get("max_pc_num", 3)),
        hidden_dim=int(config["training"].get("hidden_dim", 4096)),
        spp_levels=tuple(config["training"].get("spp_levels", (1, 2, 4))),
        pool_type=str(config["training"].get("pool_type", "avg")),
    ).to(device)

    # ---- dataset ----
    loso_json = f"loso_sbj_{fold}.json"
    base_train_ds = WeaklyHangtimeDataset(
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
        neg_keep_ratio=float(config.get("neg_keep_ratio", 0.2)),
        return_meta=True,
        seed=int(config.get("seed", 2024)) + fold,
    )

    train_dataset = ProposalWrappedDataset(
        base_ds=base_train_ds,
        num_proposals=int(config["training"]["num_proposals"]),
        backbone=backbone,
        win_len=win_len,
        seg_stride=seg_stride,
        fps=fps,
        base_physical_sec=float(config["training"].get("base_physical_sec", 7.0)),
        step_sec=float(config["training"].get("step_sec", 2.0)),
        min_sec=float(config["training"].get("min_sec", 5.0)),
        max_sec=float(config["training"].get("max_sec", 15.0)),
        seed=int(config.get("seed", 2024)) + fold,
        device=str(device),
        return_meta=False,
    )

    # ⚠️ 默认 batch_size=1（避免 mil softmax 跨 batch 混）
    bs = int(config["training"].get("batch_size", 1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )

    # ---- optim/sched ----
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 1e-5)),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"]["lr_step_size"]),
        gamma=float(config["training"]["lr_gamma"])
    )

    # ---- ckpt ----
    ckpt_dir = os.path.join(config["checkpoint_dir"], f"fold{fold}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{exp_name}.pth")

    best_loss = float("inf")
    num_epochs = int(config["training"]["num_epochs"])

    print("\n" + "=" * 80)
    print(f"[Train PCL/OICR] fold={fold} | device={device}")
    print(f"  pretrain_backbone: {pretrain_path}")
    print(f"  train windows: {len(train_dataset)} | batch={bs}")
    print("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (sample_30s, proposal_boxes, labels) in enumerate(pbar):
            sample_30s = sample_30s.to(device)          # [B,C,T]
            proposal_boxes = proposal_boxes.to(device)  # [B,P,2]
            labels = labels.to(device).float()          # [B,K]
            B = sample_30s.shape[0]

            # 提特征（冻结）
            with torch.no_grad():
                global_feat = pretrained_backbone(sample_30s)  # [B,512,Tg]

            out = model(global_feat, proposal_boxes, labels=labels)
            losses = out.get("losses", {})

            # 总损失：直接相加（你也可以加权）
            total_loss = None
            for k, v in losses.items():
                total_loss = v if total_loss is None else (total_loss + v)

            if total_loss is None:
                raise RuntimeError("Model did not return losses in training mode.")

            optimizer.zero_grad()
            total_loss.backward()

            # optional: grad clip
            if "grad_clip" in config["training"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))

            optimizer.step()

            epoch_loss += float(total_loss.item()) * B
            postfix = {"loss": f"{total_loss.item():.4f}"}
            for lk, lv in losses.items():
                postfix[lk] = f"{float(lv.item()):.3f}"
            pbar.set_postfix(postfix)

        avg_loss = epoch_loss / max(1, len(train_dataset))
        lr = scheduler.get_last_lr()[0]
        print(f"[Fold {fold}] Epoch {epoch+1} | avg_loss={avg_loss:.6f} | lr={lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict(), "best_loss": best_loss, "epoch": epoch + 1}, ckpt_path)
            print(f"  >>> saved best pcl/oicr -> {ckpt_path} (best_loss={best_loss:.6f})")

        scheduler.step()

    return ckpt_path


# ============================================================
# 4) test one fold
# ============================================================
@torch.no_grad()
def test_pcl_oicr_hangtime(config, checkpoint_path, fold: int, test_mode: str = "test_window"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 30.0))
    in_channels = int(config.get("in_channels", 113))
    num_classes = int(config["num_classes"])

    loso_json = f"loso_sbj_{fold}.json"
    ann_path = os.path.join(dataset_dir, "annotations", loso_json)
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"annotation json not found: {ann_path}")

    # dataset
    ds = WeaklyHangtimeDataset(
        dataset_dir=dataset_dir,
        loso_json=loso_json,
        mode=test_mode,
        fps=fps,
        num_sensors=in_channels,
        clip_sec=clip_sec,
        clip_overlap=float(config.get("clip_overlap", 0.0)),
        num_classes=num_classes,
        normalize=True,
        stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
        neg_keep_ratio=1.0,
        return_meta=True,
        seed=int(config.get("seed", 2024)) + fold,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(config.get("num_workers", 2)))

    # backbone + wrapper
    backbone = CNN1DBackbone(in_channels=in_channels, feat_dim=512).to(device)
    pretrain_path = os.path.join(
        config["pretrained_dir"],
        f"hangtime_{config.get('pretrained_model_name','CNN1D')}_pretrained_loso_sbj_{fold}.pth"
    )
    if not os.path.isfile(pretrain_path):
        raise FileNotFoundError(f"pretrain_path not found: {pretrain_path}")
    backbone.load_state_dict(torch.load(pretrain_path, map_location=device))
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    wrapper = GlobalBackboneWrapper(
        backbone,
        win_len=int(config.get("seg_win_len", 90)),
        seg_stride=int(config.get("seg_stride", 45)),
        chunk=int(config.get("wrapper_chunk", 256)),
    ).to(device)
    wrapper.eval()

    # model
    model = IMU_PCL_OICR(
        feat_dim=512,
        num_classes=num_classes,
        refine_times=int(config["training"].get("refine_times", 3)),
        use_pcl=bool(config["training"].get("use_pcl", False)),
        fg_thresh=float(config["training"].get("fg_thresh", 0.5)),
        bg_thresh=float(config["training"].get("bg_thresh", 0.1)),
        graph_iou_thresh=float(config["training"].get("graph_iou_thresh", 0.5)),
        max_pc_num=int(config["training"].get("max_pc_num", 3)),
        hidden_dim=int(config["training"].get("hidden_dim", 4096)),
        spp_levels=tuple(config["training"].get("spp_levels", (1, 2, 4))),
        pool_type=str(config["training"].get("pool_type", "avg")),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # id2label
    with open(ann_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    label_dict = js.get("label_dict", {})  # name->id
    id2label = {int(v): k for k, v in label_dict.items()}

    # testing params
    conf_thresh = float(config["testing"]["conf_thresh"])
    nms_sigma = float(config["testing"]["nms_sigma"])
    top_k = int(config["testing"]["top_k"])
    if test_mode == "test_window_proposals":
        num_props = int(config["testing"]["test_window_proposals"])
    else:
        num_props = int(config["testing"]["test_full_proposals"])

    fold_dir = os.path.join(config["result_root"], f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    results_cache = {}
    inf_time_list, gpu_mem_list = [], []

    for x, y, meta in tqdm(loader, desc=f"[Test PCL/OICR] fold{fold} {test_mode}"):
        sbj = str(_meta_get(meta, "sbj"))
        cs = int(_meta_get(meta, "start"))
        ce = int(_meta_get(meta, "end"))

        if sbj not in results_cache:
            results_cache[sbj] = [[] for _ in range(num_classes)]

        x = x.to(device)  # [1,C,T_frames]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        global_feat, info = wrapper(x, return_info=True)
        T_global = int(info["T_global"])
        bin_frames = int(info["bin_frames"])
        raw_frames = int(info["raw_frames"])

        props = generate_proposal_boxes(
            T_global=T_global,
            num_proposals=num_props,
            fps=fps,
            raw_frames=raw_frames,
            base_physical_sec=float(config["testing"].get("base_physical_sec", 7.0)),
            step_sec=float(config["testing"].get("step_sec", 2.0)),
            min_sec=float(config["testing"].get("min_sec", 5.0)),
            max_sec=float(config["testing"].get("max_sec", 15.0)),
            seed=int(config.get("seed", 2024)) + fold + cs,
        ).to(device)

        out = model(global_feat, props.unsqueeze(0), labels=None)

        if device.type == "cuda":
            torch.cuda.synchronize()
            gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
        inf_time_list.append((time.time() - t0) * 1000.0)

        # 用最后一层 refine（去掉背景列0）
        refine_scores = out["refine_scores"]  # list of [1,P,C+1]
        final_prob = refine_scores[-1][0, :, 1:]  # [P,C]

        # proposal -> absolute time(sec)
        for p in range(props.shape[0]):
            s_idx = int(props[p, 0].item())
            e_idx = int(props[p, 1].item())

            t_start, t_end = featbox_to_time_seconds(
                clip_start_frame=cs,
                clip_end_frame=ce,
                start_idx=s_idx,
                end_idx=e_idx,
                bin_frames=bin_frames,
                fps=fps,
            )

            for k in range(num_classes):
                score = float(final_prob[p, k].item())
                if score >= conf_thresh:
                    results_cache[sbj][k].append([t_start, t_end, score])

    # soft-nms & dump
    results = {}
    for sbj, per_cls in results_cache.items():
        final_props = []
        for k in range(num_classes):
            if len(per_cls[k]) == 0:
                continue
            segs = torch.tensor(per_cls[k], dtype=torch.float32)  # [N,3]
            segs_nms, _ = softnms_v2(segs, sigma=nms_sigma, top_k=top_k, score_threshold=conf_thresh)
            label_name = id2label.get(k, f"class_{k}")
            for seg in segs_nms:
                final_props.append({
                    "label": label_name,
                    "score": float(seg[2]),
                    "segment": [float(seg[0]), float(seg[1])]
                })
        results[sbj] = final_props

    pred_path = os.path.join(fold_dir, f"predictions_{test_mode}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump({"version": "PCL-OICR-hangtime-v1.0", "results": results, "external_data": {}},
                  f, indent=2, ensure_ascii=False)
    print(f"[Saved] {pred_path}")

    stats = {
        "fold": int(fold),
        "test_mode": str(test_mode),
        "num_samples": int(len(ds)),
        "avg_inf_time_ms": float(np.mean(inf_time_list)) if inf_time_list else 0.0,
        "std_inf_time_ms": float(np.std(inf_time_list)) if inf_time_list else 0.0,
        "avg_gpu_mem_mb": float(np.mean(gpu_mem_list)) if gpu_mem_list else 0.0,
        "std_gpu_mem_mb": float(np.std(gpu_mem_list)) if gpu_mem_list else 0.0,
        "conf_thresh": conf_thresh,
        "softnms_sigma": nms_sigma,
        "top_k": top_k,
        "num_proposals": num_props,
    }
    with open(os.path.join(fold_dir, f"inference_stats_{test_mode}.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ANET eval
    gt_path = os.path.join(fold_dir, "gt_for_anet.json")
    build_gt_for_anet(ann_path, gt_path)

    tious = np.linspace(0.3, 0.7, 5)
    evaluator = ANETdetection(
        ground_truth_filename=gt_path,
        prediction_filename=pred_path,
        subset="test",
        tiou_thresholds=tious
    )
    mAPs, avg_mAP, _ = evaluator.evaluate()

    print(f"\n[ANET] fold={fold} mode={test_mode} avg_mAP={avg_mAP:.4f}")
    for tiou, m in zip(tious, mAPs):
        print(f"  tIoU={tiou:.2f} -> mAP={m:.4f}")

    return mAPs, avg_mAP, pred_path


# ============================================================
# 5) multi-fold runner
# ============================================================
def run_loso_pcl_oicr_hangtime(config):
    set_seed(int(config.get("seed", 2024)))

    num_folds = int(config.get("num_folds", 5))
    folds = config.get("folds", list(range(num_folds)))

    os.makedirs(config["result_root"], exist_ok=True)

    all_reports = []

    for i, fold in enumerate(folds):
        print("\n" + "=" * 90)
        print(f"[LOSO/KFold] fold={fold} ({i+1}/{len(folds)})")
        print("=" * 90)

        ckpt = train_pcl_oicr_one_fold_hangtime(
            config, fold, exp_name=config.get("exp_name", "pcl_oicr_opportunity")
        )

        mAPs_w, avg_w, pred_w = test_pcl_oicr_hangtime(config, ckpt, fold=fold, test_mode="test_window")
        mAPs_f, avg_f, pred_f = test_pcl_oicr_hangtime(config, ckpt, fold=fold, test_mode="test_full")

        all_reports.append({
            "fold": int(fold),
            "ckpt": ckpt,
            "test_window": {
                "pred_path": pred_w,
                "tious": [float(x) for x in np.linspace(0.3, 0.7, 5)],
                "mAPs": [float(x) for x in mAPs_w],
                "avg_mAP": float(avg_w),
            },
            "test_full": {
                "pred_path": pred_f,
                "tious": [float(x) for x in np.linspace(0.3, 0.7, 5)],
                "mAPs": [float(x) for x in mAPs_f],
                "avg_mAP": float(avg_f),
            },
        })

        with open(os.path.join(config["result_root"], "loso_report_partial.json"), "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] report -> {all_reports}")


# ============================================================
# 6) main
# ============================================================
if __name__ == "__main__":
    config = {
        "seed": 2024,
        "exp_name": "oicr_hangtime",

        "dataset_dir": "/home/lipei/TAL_data/hangtime/",
        "pretrained_dir": "/home/lipei/project/WSDDN/HANGTIME/pre_train",
        "checkpoint_dir": "/home/lipei/project/WSDDN/checkpoints/HANGTIME/oicr_0106",
        "result_root": "/home/lipei/project/WSDDN/test_results/HANGTIME/oicr_0106",

        "num_folds": 24,
        "folds": [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],

        "fps": 50,
        "clip_sec": 30.0,
        "clip_overlap": 0.5,
        "in_channels": 3,
        "num_classes": 5,
        "stats_dirname": "loso_norm_stats_json",

        "seg_win_len": 90,
        "seg_stride": 45,
        "wrapper_chunk": 256,

        "pretrained_model_name": "CNN1D",

        "num_workers": 4,
        "neg_keep_ratio": 0.2,

        "training": {
            "batch_size": 16,
            "num_epochs": 60,
            "lr": 1e-4,
            "lr_step_size": 20,
            "lr_gamma": 0.5,
            "weight_decay": 1e-5,
            # "grad_clip": 5.0,

            "num_proposals": 80,

            # proposal params
            "base_physical_sec": 3.0,
            "step_sec": 2.0,
            "min_sec": 1.0,
            "max_sec": 30.0,

            # PCL/OICR params
            "refine_times": 3,
            "use_pcl": False,          # True=PCL, False=OICR
            "fg_thresh": 0.5,
            "bg_thresh": 0.1,
            "graph_iou_thresh": 0.5,
            "max_pc_num": 3,
            "hidden_dim": 4096,
            "spp_levels": (1, 2, 4),
            "pool_type": "avg",
        },

        "testing": {
            "test_window_proposals": 100,
            "test_full_proposals": 3000,
            "conf_thresh": 0.0,
            "nms_sigma": 0.5,
            "top_k": 200,

            "base_physical_sec": 3.0,
            "step_sec": 2.0,
            "min_sec": 1.0,
            "max_sec": 30.0,
        }
    }

    run_loso_pcl_oicr_hangtime(config)
