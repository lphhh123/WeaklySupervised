
import torch.optim as optim
from tqdm import tqdm

from OtherData.utils import _meta_get
from models.WSDDN_model import WSDDN
from tool import softnms_v2, ANETdetection
from OtherData.Opportunity.dataset_opportunity_ws import WeaklyOpportunityDataset
from pre_train.pre_model import CNN1DBackbone
from OtherData.utils import *

# ============================================================
# train one fold
# ============================================================
def train_wsddn_one_fold_opportunity(config, fold: int, exp_name: str = "wsddn_opportunity"):
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
        f"opportunity_{config.get('pretrained_model_name','CNN1D')}_pretrained_loso_sbj_{fold}.pth"
    )
    backbone.load_state_dict(torch.load(pretrain_path, map_location=device))
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # ---- pretrained_backbone ----
    win_len = int(config.get("seg_win_len", 90))
    seg_stride = int(config.get("seg_stride", 45))
    pretrained_backbone = GlobalBackboneWrapper(backbone, win_len=win_len, seg_stride=seg_stride).to(device)
    pretrained_backbone.eval()

    # ---- wsddn ----
    model = WSDDN(num_classes=num_classes, feat_dim=512).to(device)

    # ---- dataset ----
    loso_json = f"loso_sbj_{fold}.json"
    base_train_ds = WeaklyOpportunityDataset(
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
        num_proposals=config["training"]["num_proposals"],
        backbone=backbone,  # 关键：传进来probe Lout
        win_len=config.get("seg_win_len", 90),
        seg_stride=config.get("seg_stride", 45),
        fps=config.get("fps", 30),
        base_physical_sec=config["training"].get("base_physical_sec", 7.0),
        step_sec=config["training"].get("step_sec", 2.0),
        min_sec=config["training"].get("min_sec", 5.0),
        max_sec=config["training"].get("max_sec", 15.0),
        seed=config.get("seed", 2024) + fold,
        device=device,
        return_meta=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )

    # ---- optim/sched/criterion ----
    optimizer = optim.Adam(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"]["lr_step_size"]),
        gamma=float(config["training"]["lr_gamma"])
    )
    criterion = nn.BCELoss()

    lambda_reg = float(config["training"].get("spatial_reg_weight", 0.0))
    iou_thresh = float(config["training"].get("spatial_reg_iou", 0.8))

    # ---- ckpt ----
    ckpt_dir = os.path.join(config["checkpoint_dir"], f"fold{fold}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{exp_name}.pth")

    best_loss = float("inf")
    num_epochs = int(config["training"]["num_epochs"])

    print("\n" + "=" * 80)
    print(f"[Train] fold={fold} | device={device}")
    print(f"  pretrain_backbone: {pretrain_path}")
    print(f"  train windows: {len(train_dataset)} | batch={config['training']['batch_size']}")
    print("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (sample_30s, proposal_boxes, labels) in enumerate(pbar):
            sample_30s = sample_30s.to(device)         # [B,C,900]
            proposal_boxes = proposal_boxes.to(device) # [B,P,2] or [P,2]
            labels = labels.to(device).float()         # [B,K] multi-hot
            B = sample_30s.shape[0]

            if proposal_boxes.ndim == 2:
                proposal_boxes = proposal_boxes.unsqueeze(0).repeat(B, 1, 1)


            global_feat = pretrained_backbone(sample_30s)  # [B,512,T_global]
            outputs = model(global_feat, proposal_boxes)

            video_prob = outputs["video_prob"]  # [B,K]
            joint_prob = outputs["joint_prob"]  # [B,P,K]
            feat_fc7 = outputs["feat_fc7"]      # [B,P,D]

            video_prob_safe = torch.clamp(video_prob, 1e-6, 1.0 - 1e-6)
            bce_loss = criterion(video_prob_safe, labels)

            # ---- spatial regularizer ----
            spatial_reg = torch.tensor(0.0, device=device)
            reg_count = 0
            if lambda_reg > 0.0:
                for b in range(B):
                    pos_cls = (labels[b] > 0).nonzero(as_tuple=False).view(-1)
                    if pos_cls.numel() == 0:
                        continue

                    boxes_b = proposal_boxes[b].float()  # [P,2]
                    feats_b = feat_fc7[b]               # [P,D]

                    for cls_idx in pos_cls:
                        cls_idx = int(cls_idx.item())
                        scores_bc = joint_prob[b, :, cls_idx]  # [P]
                        p_star = int(scores_bc.argmax().item())

                        start_star, end_star = boxes_b[p_star][0], boxes_b[p_star][1]
                        start_all = boxes_b[:, 0]
                        end_all = boxes_b[:, 1]

                        tt1 = torch.maximum(start_all, start_star)
                        tt2 = torch.minimum(end_all, end_star)
                        inter = (tt2 - tt1).clamp(min=0)

                        len_all = (end_all - start_all).clamp(min=1e-6)
                        len_star = (end_star - start_star).clamp(min=1e-6)
                        union = len_all + len_star - inter
                        iou = inter / union

                        idx_all = torch.arange(boxes_b.size(0), device=device)
                        neighbor_mask = (iou > iou_thresh) & (idx_all != p_star)

                        if neighbor_mask.any():
                            f_star = feats_b[p_star].unsqueeze(0)  # [1,D]
                            f_nb = feats_b[neighbor_mask]          # [K,D]
                            reg_term = (f_nb - f_star).pow(2).mean()
                            spatial_reg = spatial_reg + reg_term
                            reg_count += 1

            if lambda_reg > 0.0 and reg_count > 0:
                spatial_reg = spatial_reg / reg_count
                total_loss = bce_loss + lambda_reg * spatial_reg
            else:
                total_loss = bce_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * B
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "bce": f"{bce_loss.item():.4f}",
                "reg": f"{float(spatial_reg.item()) if reg_count>0 else 0.0:.4f}"
            })

        avg_loss = epoch_loss / max(1, len(train_dataset))
        lr = scheduler.get_last_lr()[0]
        print(f"[Fold {fold}] Epoch {epoch+1} | avg_loss={avg_loss:.6f} | lr={lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict(), "best_loss": best_loss, "epoch": epoch + 1}, ckpt_path)
            print(f"  >>> saved best wsddn -> {ckpt_path} (best_loss={best_loss:.6f})")

        scheduler.step()

    return ckpt_path


# ============================================================
# 5) test one fold
# ============================================================
@torch.no_grad()
def test_wsddn_opportunity(config, checkpoint_path, fold: int, test_mode: str = "test_window"):
    """
    test_mode:
      - "test_window": 测试人也按 clip_sec 滑窗（最贴近训练）
      - "test_full"  : 整条序列一次性跑（需要你的 dataset 支持该 mode）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 30.0))
    in_channels = int(config.get("in_channels", 113))
    num_classes = int(config["num_classes"])

    loso_json = f"loso_sbj_{fold}.json"
    ann_path = os.path.join(dataset_dir, "annotations", loso_json)

    # ---- dataset ----
    ds = WeaklyOpportunityDataset(
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

    # ---- load backbone + wrapper（与训练一致）----
    backbone = CNN1DBackbone(in_channels=in_channels, feat_dim=512).to(device)

    pretrain_path = os.path.join(
        config["pretrained_dir"],
        f"opportunity_{config.get('pretrained_model_name','CNN1D')}_pretrained_loso_sbj_{fold}.pth"
    )
    backbone.load_state_dict(torch.load(pretrain_path, map_location=device))
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    wrapper = GlobalBackboneWrapper(
        backbone,
        win_len=int(config.get("seg_win_len", 90)),
        seg_stride=int(config.get("seg_stride", 45)),
        chunk=256
    ).to(device)
    wrapper.eval()

    # ---- load wsddn ----
    wsddn = WSDDN(num_classes=num_classes, feat_dim=512).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        wsddn.load_state_dict(ckpt["model_state_dict"])
    else:
        wsddn.load_state_dict(ckpt)
    wsddn.eval()

    # ---- id2label from loso json ----
    with open(ann_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    label_dict = js.get("label_dict", {})  # name->id
    id2label = {int(v): k for k, v in label_dict.items()}

    # ---- testing params ----
    conf_thresh = float(config["testing"]["conf_thresh"])
    nms_sigma = float(config["testing"]["nms_sigma"])
    top_k = int(config["testing"]["top_k"])
    if test_mode == "test_window":
        num_props = int(config["testing"]["test_window_proposals"])
    else:
        num_props = config["testing"]["test_full_proposals"]

    # ---- output dir ----
    fold_dir = os.path.join(config["result_root"], f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # results_cache[sbj][k] = [[t0,t1,score], ...]
    results_cache = {}
    inf_time_list, gpu_mem_list = [], []

    for x, y, meta in tqdm(loader, desc=f"[Test] fold{fold} {test_mode}"):
        sbj = str(_meta_get(meta, "sbj"))
        cs = int(_meta_get(meta, "start"))  # frame index
        ce = int(_meta_get(meta, "end"))    # frame index

        if sbj not in results_cache:
            results_cache[sbj] = [[] for _ in range(num_classes)]

        x = x.to(device)  # [1,C,T_frames]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        global_feat, info = wrapper(x, return_info=True)  # [1,512,T_global]
        T_global = int(info["T_global"])
        bin_frames = int(info["bin_frames"])
        raw_frames = int(info["raw_frames"])

        props = generate_proposal_boxes(
            T_global=T_global,
            num_proposals=num_props,
            fps=fps,
            raw_frames=raw_frames,  # test_full 时长度可变，用 raw_frames 更稳
            base_physical_sec=float(config["testing"].get("base_physical_sec", 7.0)),
            step_sec=float(config["testing"].get("step_sec", 2.0)),
            min_sec=float(config["testing"].get("min_sec", 5.0)),
            max_sec=float(config["testing"].get("max_sec", 15.0)),
            seed=int(config.get("seed", 2024)) + fold + cs
        ).to(device)

        out = wsddn(global_feat, props.unsqueeze(0))

        if device.type == "cuda":
            torch.cuda.synchronize()
            gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
        inf_time_list.append((time.time() - t0) * 1000.0)

        joint_prob = out["joint_prob"][0]  # [P,K]

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
                score = float(joint_prob[p, k].item())
                if score >= conf_thresh:
                    results_cache[sbj][k].append([t_start, t_end, score])

    # ---- soft-nms & dump predictions ----
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
        json.dump({"version": "WSDDN-Opportunity-v1.0", "results": results, "external_data": {}},
                  f, indent=2, ensure_ascii=False)
    print(f"[Saved] {pred_path}")

    # ---- stats ----
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

    # ---- ANET eval ----
    gt_path = os.path.join(fold_dir, "gt_for_anet.json")
    build_gt_for_anet(ann_path, gt_path)

    tious = np.linspace(0.3, 0.7, 5)
    evaluator = ANETdetection(
        ground_truth_filename=gt_path,
        prediction_filename=pred_path,
        subset="test",
        tiou_thresholds=tious
    )
    mAPs, avg_mAP, ap_mat  = evaluator.evaluate()

    # -----------------------------
    # 保存每个动作(per-class)的 AP 到 json
    # ap_mat: [len(tious), num_classes]
    # -----------------------------
    idx2name = {int(v): str(k) for k, v in evaluator.activity_index.items()}  # idx -> label_name

    per_action = {}
    # 为了稳定输出顺序：按 idx 从小到大写
    for cidx in range(ap_mat.shape[1]):
        name = idx2name.get(cidx, id2label.get(cidx, f"class_{cidx}"))
        ap_list = [float(x) for x in ap_mat[:, cidx].tolist()]  # 每个 tIoU 的 AP
        per_action[name] = {
            "ap_per_tiou": ap_list,
            "mean_ap": float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0
        }

    per_action_path = os.path.join(fold_dir, f"per_action_ap_{test_mode}.json")
    with open(per_action_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fold": int(fold),
                "test_mode": str(test_mode),
                "tious": [float(x) for x in tious],
                "avg_mAP": float(avg_mAP),
                "mAP_per_tiou": [float(x) for x in mAPs],
                "per_action": per_action
            },
            f,
            indent=2,
            ensure_ascii=False
        )
    print(f"[Saved] {per_action_path}")

    print(f"\n[ANET] fold={fold} mode={test_mode} avg_mAP={avg_mAP:.4f}")
    for tiou, m in zip(tious, mAPs):
        print(f"  tIoU={tiou:.2f} -> mAP={m:.4f}")

    return mAPs, avg_mAP, pred_path


# ============================================================
# 6) multi-fold runner
# ============================================================
def run_loso_wsddn_opportunity(config):
    set_seed(int(config.get("seed", 2024)))

    num_folds = int(config.get("num_folds", 5))
    folds = config.get("folds", list(range(num_folds)))

    os.makedirs(config["result_root"], exist_ok=True)

    all_reports = []

    for i, fold in enumerate(folds):
        print("\n" + "=" * 90)
        print(f"[LOSO/KFold] fold={fold} ({i+1}/{len(folds)})")
        print("=" * 90)

        # 1) train
        wsddn_ckpt = train_wsddn_one_fold_opportunity(config, fold, exp_name=config.get("exp_name", "wsddn_opportunity"))

        # 2) test_window
        mAPs_w, avg_w, pred_w = test_wsddn_opportunity(config, wsddn_ckpt, fold=fold, test_mode="test_window")

        # 3) test_full（如果你的 dataset 支持）
        mAPs_f, avg_f, pred_f = test_wsddn_opportunity(config, wsddn_ckpt, fold=fold, test_mode="test_full")

        all_reports.append({
            "fold": int(fold),
            "wsddn_ckpt": wsddn_ckpt,
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

        # 每折落盘一次
        with open(os.path.join(config["result_root"], "loso_report_partial.json"), "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)

    report_path = os.path.join(config["result_root"], "loso_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] report -> {report_path}")


# ============================================================
# 7) main
# ============================================================
if __name__ == "__main__":
    config = {
        "seed": 2024,
        "exp_name": "wsddn_opportunity",

        "dataset_dir": "/home/lipei/TAL_data/opportunity/",
        "pretrained_dir": "/home/lipei/project/WSDDN/Opportunity/pre_train",
        "checkpoint_dir": "/home/lipei/project/WSDDN/checkpoints/Opportunity/wsddn_0105",
        "result_root": "/home/lipei/project/WSDDN/test_results/Opportunity/wsddn_0105",

        "num_folds": 4,
        "folds": [0, 1, 2, 3],  # 只跑部分折就改这里

        # Opportunity data
        "fps": 30,
        "clip_sec": 30.0,
        "clip_overlap": 0.5,
        "in_channels": 113,        # 传感器轴数
        "num_classes": 17,
        "stats_dirname": "loso_norm_stats_json",

        # wrapper params
        "seg_win_len": 90,         # 3s*30
        "seg_stride": 45,          # 1.5s*30

        "pretrained_model_name": "CNN1D",

        "num_workers": 4,

        "neg_keep_ratio": 0.1,

        "training": {
            "batch_size": 16,
            "num_epochs": 60,
            "lr": 1e-4,
            "lr_step_size": 20,
            "lr_gamma": 0.5,

            "num_proposals": 80,


            "base_physical_sec": 3.0,
            "step_sec": 2.0,
            "min_sec": 1.0,
            "max_sec": 17.0,

            # spatial regularizer（可为0关闭）
            "spatial_reg_weight": 1.0,
            "spatial_reg_iou": 0.8,
        },

        "testing": {
            "test_window_proposals":100,
            "test_full_proposals": 2000,
            "conf_thresh": 0.0,
            "nms_sigma": 0.5,
            "top_k": 200,

            # proposal params（测试可放宽/修改）
            "base_physical_sec": 3.0,
            "step_sec": 2.0,
            "min_sec": 1.0,
            "max_sec": 17.0,
        }
    }

    run_loso_wsddn_opportunity(config)
