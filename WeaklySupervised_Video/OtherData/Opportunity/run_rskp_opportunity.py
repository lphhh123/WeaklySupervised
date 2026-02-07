import gc
import os, json, time
import argparse
import numpy as np

                                                    
import sys

from scipy import ndimage

if "--gpu" in sys.argv:
    gpu_idx = sys.argv[sys.argv.index("--gpu") + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace

from OtherData.Opportunity.dataset_opportunity_ws import WeaklyOpportunityDataset
from OtherData.utils import _meta_get, set_seed, featbox_to_time_seconds, build_gt_for_anet
from OtherData.utils import GlobalBackboneWrapper, ProposalWrappedDataset, dump_config
from pre_train.pre_model import CNN1DBackbone
from tool import softnms_v2, ANETdetection

         
from RSKP_MODEL.main_branch import WSTAL, random_walk
from RSKP_MODEL.memory import Memory
from RSKP_MODEL.losses import NormalizedCrossEntropy, AttLoss, CategoryCrossEntropy



def count_parameters(model):
    """
    Count the number of model parameters.
    Args:
        model: PyTorch model
    Returns:
        total_params: total number of model parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params



# ============================================================
# train one fold
# ============================================================
def train_rskp_one_fold_opportunity(config, fold: int, exp_name: str = "rskp_opportunity", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = config["dataset_dir"]
    fps = int(config.get("fps", 30))
    clip_sec = float(config.get("clip_sec", 30.0))
    in_channels = int(config.get("in_channels", 113))
    num_classes = int(config["num_classes"])

             
    rskp_cfg = config.get("rskp", {})

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

                      
    dummy_input = torch.randn(1, in_channels, win_len, device=device)
    with torch.no_grad():
        dummy_feat = pretrained_backbone(dummy_input)
    feat_dim = int(dummy_feat.shape[1])  # 512

                       
    out_feat_num = rskp_cfg.get("out_feat_num", feat_dim)
    d_model = int(out_feat_num) if out_feat_num is not None else feat_dim
    args = SimpleNamespace(
        w=float(rskp_cfg.get("w", 0.5)),
        inp_feat_num=d_model,
        out_feat_num=d_model,
        mu_num=int(rskp_cfg.get("mu_num", 8)),
        em_iter=int(rskp_cfg.get("em_iter", 2)),
        class_num=num_classes,
        scale_factor=float(rskp_cfg.get("scale_factor", 20.0)),
        dropout=float(rskp_cfg.get("dropout", 0.6)),
        mu_queue_len=int(rskp_cfg.get("mu_queue_len", 5)),
    )

    model = WSTAL(args).to(device)
    memory = Memory(args).to(device)
    
    
       
            
    model_params = count_parameters(model)
    memory_params = count_parameters(memory)
    total_params = model_params + memory_params
    print(f"  Model parameters: {model_params:,} ({model_params/1e6:.2f}M)")
    print(f"  Memory parameters: {memory_params:,} ({memory_params/1e6:.2f}M)")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    loss_att = AttLoss(8.0)
    loss_nce = NormalizedCrossEntropy()
    loss_spl = CategoryCrossEntropy(float(rskp_cfg.get("T", 0.2)))

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
        seed=int(config.get("seed", 2024)),
    )
    train_dataset = ProposalWrappedDataset(
        base_ds=base_train_ds,
        num_proposals=config["training"]["num_proposals"],
        backbone=backbone,
        win_len=config.get("seg_win_len", 90),
        seg_stride=config.get("seg_stride", 45),
        fps=config.get("fps", 30),
        base_physical_sec=config["training"].get("base_physical_sec", 7.0),
        step_sec=config["training"].get("step_sec", 2.0),
        min_sec=config["training"].get("min_sec", 5.0),
        max_sec=config["training"].get("max_sec", 15.0),
        seed=config.get("seed", 2024),
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

    # ---- optim/sched ----
    optimizer = optim.Adam(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"]["lr_step_size"]),
        gamma=float(config["training"]["lr_gamma"])
    )

               
    warmup_epoch = int(rskp_cfg.get("warmup_epoch", 1))
    lambda_a = float(rskp_cfg.get("lambda_a", 0.1))
    lambda_b = float(rskp_cfg.get("lambda_b", 0.2))
    lambda_s = float(rskp_cfg.get("lambda_s", 1.0))

    # ---- ckpt ----
    ckpt_dir = os.path.join(config["checkpoint_dir"], f"fold{fold}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{exp_name}.pth")

    best_loss = float("inf")
    num_epochs = int(config["training"]["num_epochs"])

    print("\n" + "=" * 80)
    print(f"[RSKP-Train] fold={fold} | device={device}")
    print(f"  pretrain_backbone: {pretrain_path}")
    print(f"  train windows: {len(train_dataset)} | batch={config['training']['batch_size']}")
    print(f"  warmup_epoch={warmup_epoch}, lambda_a={lambda_a}, lambda_b={lambda_b}, lambda_s={lambda_s}")
    print("=" * 80)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        loss_recorder = {"cls_fore": 0.0, "cls_back": 0.0, "att": 0.0, "spl": 0.0}

        pbar = tqdm(train_loader, desc=f"[RSKP Fold {fold}] Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (sample_30s, proposal_boxes, labels) in enumerate(pbar):
            sample_30s = sample_30s.to(device)         # [B,C,900]
            labels = labels.to(device).float()         # [B,K] multi-hot
            B = sample_30s.shape[0]

                                         
            with torch.no_grad():
                global_feat = pretrained_backbone(sample_30s)  # [B,512,T_global]
            feat_seq = global_feat.permute(0, 2, 1)  # [B, T, D]

            o_out, m_out, em_out = model(feat_seq)

            f_labels = torch.cat([labels, torch.zeros(labels.size(0), 1, device=device)], dim=1)
            b_labels = torch.cat([labels, torch.ones(labels.size(0), 1, device=device)], dim=1)

            vid_fore_loss = loss_nce(o_out[0], f_labels) + loss_nce(m_out[0], f_labels)
            vid_back_loss = loss_nce(o_out[1], b_labels) + loss_nce(m_out[1], b_labels)
            vid_att_loss = loss_att(o_out[2])

                                         
            if epoch >= warmup_epoch:
                np_labels = labels.detach().cpu().numpy()
                idxs = []
                for b in range(B):
                    pos_cls = np.where(np_labels[b] == 1)[0].tolist()
                    idxs.extend(pos_cls)
                idxs = list(set(idxs))      

                if len(idxs) > 0:
                    cls_mu = memory._return_queue(idxs).detach()  # [1, num_classes, feat_dim]
                    cls_mu = cls_mu.expand(B, -1, -1)  # [B, num_classes, feat_dim]
                    reallocated_x = random_walk(em_out[0], cls_mu, args.w)
                    r_vid_ca_pred, r_vid_cw_pred, _, r_frm_pred = model.PredictionModule(reallocated_x)

                    vid_fore_loss = vid_fore_loss + 0.5 * loss_nce(r_vid_ca_pred, f_labels)
                    vid_back_loss = vid_back_loss + 0.5 * loss_nce(r_vid_cw_pred, b_labels)
                    vid_spl_loss = loss_spl(o_out[3], r_frm_pred * 0.2 + m_out[3] * 0.8)

                                 
                    for b in range(B):
                        mu = em_out[1][b]  # [mu_num, feat_dim]
                        mu_pred = em_out[2][b]  # [mu_num, num_classes+1]
                        pos_cls_b = np.where(np_labels[b] == 1)[0].tolist()
                        
                        for cls_idx in pos_cls_b:
                            sc = mu_pred[:, cls_idx]  # [mu_num]
                            memory._update_queue(mu, sc, [cls_idx])
                else:
                    vid_spl_loss = loss_spl(o_out[3], m_out[3])
            else:
                vid_spl_loss = loss_spl(o_out[3], m_out[3])

            total_loss = vid_fore_loss + lambda_b * vid_back_loss + lambda_a * vid_att_loss + lambda_s * vid_spl_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * B
            loss_recorder["cls_fore"] += vid_fore_loss.item()
            loss_recorder["cls_back"] += vid_back_loss.item()
            loss_recorder["att"] += vid_att_loss.item()
            loss_recorder["spl"] += vid_spl_loss.item()

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "fore": f"{vid_fore_loss.item():.4f}",
                "back": f"{vid_back_loss.item():.4f}",
                "att": f"{vid_att_loss.item():.4f}",
                "spl": f"{vid_spl_loss.item():.4f}"
            })

        avg_loss = epoch_loss / max(1, len(train_dataset))
        lr = scheduler.get_last_lr()[0]
        print(f"[RSKP Fold {fold}] Epoch {epoch+1} | avg_loss={avg_loss:.6f} | lr={lr:.6f}")
        print(f"  Loss breakdown: {loss_recorder}")

                                  
        if epoch == warmup_epoch:
            print(f"  >>> Initializing memory queue at epoch {epoch+1}...")
            model.eval()
            
                                
            mu_queue = []
            sc_queue = []
            lbl_queue = []
            
            for sample_30s, proposal_boxes, labels in train_loader:
                sample_30s = sample_30s.to(device)
                labels = labels.to(device).float()
                
                with torch.no_grad():
                    global_feat = pretrained_backbone(sample_30s)
                    feat_seq = global_feat.permute(0, 2, 1)
                    
                    _, _, em_out = model(feat_seq)
                    
                    mu_queue.append(em_out[1][0].detach().cpu())  # [mu_num, feat_dim]
                    sc_queue.append(em_out[2][0].detach().cpu())  # [mu_num, num_classes+1]
                    
                    lbl = labels.detach().cpu().numpy()
                    lbl_queue.append(lbl)  # [num_classes]
            
                    
            memory._init_queue(mu_queue, sc_queue, lbl_queue)
            model.train()
            
                         
            lambda_s = 0.5
            print(f"  >>> lambda_s adjusted to {lambda_s}")
            
                                          
            best_loss = float("inf")
            print(f"  >>> Reset best_loss due to loss computation change")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "memory_state_dict": memory.state_dict(),
                "best_loss": best_loss,
                "epoch": epoch + 1,
                "args": vars(args),
                "num_classes": num_classes,
            }, ckpt_path)
            print(f"  >>> saved best RSKP -> {ckpt_path} (best_loss={best_loss:.6f})")

        scheduler.step()

    return ckpt_path


# ============================================================
# test one fold
# ============================================================
@torch.no_grad()
def test_rskp_opportunity(config, checkpoint_path, fold: int, test_mode: str = "test_window", device=None):
    """
    test_mode:
      - "test_window": test subjects also use clip_sec sliding windows
      - "test_full"  : run the full sequence in one pass
    """
    if device is None:
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
        seed=int(config.get("seed", 2024)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(config.get("num_workers", 2)))

    # ---- load backbone + wrapper ----
    backbone = CNN1DBackbone(in_channels=in_channels, feat_dim=512).to(device)

    pretrain_path = os.path.join(
        config["pretrained_dir"],
        f"opportunity_{config.get('pretrained_model_name','CNN1D')}_pretrained_loso_sbj_{fold}.pth"
    )
    backbone.load_state_dict(torch.load(pretrain_path, map_location=device))
    backbone.eval()

    wrapper = GlobalBackboneWrapper(
        backbone,
        win_len=int(config.get("seg_win_len", 90)),
        seg_stride=int(config.get("seg_stride", 45)),
        chunk=256
    ).to(device)
    wrapper.eval()

    # ---- load RSKP model ----
    ckpt = torch.load(checkpoint_path, map_location=device)
    args_dict = ckpt.get("args", {})
    args = SimpleNamespace(
        w=args_dict.get("w", 0.5),
        inp_feat_num=args_dict.get("inp_feat_num", 512),
        out_feat_num=args_dict.get("out_feat_num", 512),
        mu_num=args_dict.get("mu_num", 8),
        em_iter=args_dict.get("em_iter", 2),
        class_num=num_classes,
        scale_factor=args_dict.get("scale_factor", 20.0),
        dropout=args_dict.get("dropout", 0.6),
        mu_queue_len=args_dict.get("mu_queue_len", 5),
    )

    model = WSTAL(args).to(device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
        
            
    model_params = count_parameters(model)
    print(f"  Model parameters: {model_params:,} ({model_params/1e6:.2f}M)")

    # ---- id2label from loso json ----
    with open(ann_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    label_dict = js.get("label_dict", {})
    id2label = {int(v): k for k, v in label_dict.items()}

    # ---- testing params ----
    conf_thresh = float(config["testing"]["conf_thresh"])
    nms_sigma = float(config["testing"]["nms_sigma"])
    top_k = int(config["testing"]["top_k"])
    rskp_cfg = config.get("rskp", {})
              
    test_fusion_weight = float(rskp_cfg.get("test_fusion_weight", 0.6))
    test_reallocated_weight = 1.0 - test_fusion_weight

    # ---- output dir ----
    fold_dir = os.path.join(config["result_root"], f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # results_cache[sbj][k] = [[t0,t1,score], ...]
    results_cache = {}
    inf_time_list, gpu_mem_list = [], []

    for x, y, meta in tqdm(loader, desc=f"[RSKP Test] fold{fold} {test_mode}"):
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

        global_feat, info = wrapper(x, return_info=True)  # [1,512,T_global]
        T_global = int(info["T_global"])
        bin_frames = int(info["bin_frames"])
        raw_frames = int(info["raw_frames"])

        feat_seq = global_feat.permute(0, 2, 1)  # [1, T, D]
        o_out, m_out, _ = model(feat_seq)

                           
        vid_pred = o_out[0] * test_fusion_weight + m_out[0] * test_reallocated_weight  # [1, C]
        frm_pred = torch.softmax(o_out[3], -1) * test_fusion_weight + torch.softmax(m_out[3], -1) * test_reallocated_weight  # [1, T, C]
        vid_att = o_out[2]  # [1, T]
        frm_pred = frm_pred * vid_att[..., None]         

        if device.type == "cuda":
            torch.cuda.synchronize()
            gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
        inf_time_list.append((time.time() - t0) * 1000.0)

        frm_pred = frm_pred.squeeze(0).cpu()  # [T, C]
        T_seq = frm_pred.shape[0]

                      
        for k in range(num_classes):
            scores = frm_pred[:, k].numpy()
            scores = ndimage.gaussian_filter1d(scores, sigma=2.0)
            above = scores >= conf_thresh
            if not above.any():
                continue

                       
            start_idx = None
            for t, flag in enumerate(above):
                if flag and start_idx is None:
                    start_idx = t
                if (not flag or t == T_seq - 1) and start_idx is not None:
                    end_t = t if not flag else t + 1
                    t_start, t_end = featbox_to_time_seconds(
                        clip_start_frame=cs,
                        clip_end_frame=ce,
                        start_idx=start_idx,
                        end_idx=end_t,
                        bin_frames=bin_frames,
                        fps=fps,
                    )
                    segment_score = float(scores[start_idx:end_t].max())
                    results_cache[sbj][k].append([t_start, t_end, segment_score])
                    start_idx = None

    # ---- soft-nms & dump predictions ----
    results = {}
    for sbj, per_cls in results_cache.items():
        final_props = []
        for k in range(num_classes):
            if len(per_cls[k]) == 0:
                continue
            segs = torch.tensor(per_cls[k], dtype=torch.float32)
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
        json.dump({"version": "RSKP-opportunity-v1.0", "results": results, "external_data": {}},
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
    mAPs, avg_mAP, ap_mat = evaluator.evaluate()

                
    idx2name = {int(v): str(k) for k, v in evaluator.activity_index.items()}
    per_action = {}
    for cidx in range(ap_mat.shape[1]):
        name = idx2name.get(cidx, id2label.get(cidx, f"class_{cidx}"))
        ap_list = [float(x) for x in ap_mat[:, cidx].tolist()]
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

    print(f"\n[RSKP-ANET] fold={fold} mode={test_mode} avg_mAP={avg_mAP:.4f}")
    for tiou, m in zip(tious, mAPs):
        print(f"  tIoU={tiou:.2f} -> mAP={m:.4f}")

    return mAPs, avg_mAP, pred_path


# ============================================================
# multi-fold runner
# ============================================================
def run_loso_rskp_opportunity(config, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(int(config.get("seed", 2024)))
    os.makedirs(config["result_root"], exist_ok=True)
    dump_config(config, config["result_root"])

    num_folds = int(config.get("num_folds", 5))
    folds = config.get("folds", list(range(num_folds)))

    all_reports = []

    for i, fold in enumerate(folds):
        print("\n" + "=" * 90)
        print(f"[LOSO/KFold] fold={fold} ({i+1}/{len(folds)})")
        print("=" * 90)

        # 1) train
        rskp_ckpt = train_rskp_one_fold_opportunity(config, fold, exp_name=config.get("exp_name", "rskp_opportunity"), device=device)

        # 2) test_window
        mAPs_w, avg_w, pred_w = test_rskp_opportunity(config, rskp_ckpt, fold=fold, test_mode="test_window", device=device)

        # 3) test_full
        mAPs_f, avg_f, pred_f = test_rskp_opportunity(config, rskp_ckpt, fold=fold, test_mode="test_full", device=device)

        all_reports.append({
            "fold": int(fold),
            "rskp_ckpt": rskp_ckpt,
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

              
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print(f"[Memory Cleaned] fold={fold} completed")

    print(f"\n[Saved] report -> {all_reports}")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSKP WETLAB Training")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID to use (e.g., 0, 1, 2)")
    parser.add_argument("--seed", type=int, default=2024, help="seed (e.g., 0, 1, 2)")
    args = parser.parse_args()

    if args.gpu is not None:
        device = torch.device(f"cuda:0")
    else:
        device = None

    config = {
        "seed": args.seed,
        "exp_name": "rskp_opportunity",

        "dataset_dir": "/home/lipei/TAL_data/opportunity/",
        "pretrained_dir": "/home/lipei/project/WSDDN/OtherData/Opportunity/pre_train/CNN1D",
        "checkpoint_dir": f"/home/lipei/project/WSDDN/checkpoints/Opportunity/{args.seed}/rskp_0130",
        "result_root": f"/home/lipei/project/WSDDN/test_results/Opportunity/{args.seed}/rskp_0130",

        "num_folds": 4,
        "folds": [0, 1, 2, 3],

        "fps": 30,
        "clip_sec": 3.0,
        "clip_overlap": 0.5,
        "in_channels": 113,
        "num_classes": 17,
        "stats_dirname": "loso_norm_stats_json",

        "seg_win_len": 90,
        "seg_stride": 45,
        "pretrained_model_name": "CNN1D",
        "num_workers": 4,
        "neg_keep_ratio": 0.1,

        "training": {
            "batch_size": 32,
            "num_epochs": 80,
            "lr": 1e-4,
            "lr_step_size": 10,
            "lr_gamma": 0.9,
            "num_proposals": 80,
            "base_physical_sec": 3.0,
            "step_sec": 1.0,
            "min_sec": 1.0,
            "max_sec": 17.0,
        },

        "testing": {
            "conf_thresh": 0.02,
            "nms_sigma": 0.5,
            "top_k": 200,
        },

                 
        "rskp": {
            "out_feat_num": None,                          
            "w": 0.2,
            "mu_num": 8,
            "em_iter": 3,
            "mu_queue_len": 8,
            "scale_factor": 4.0,
            "dropout": 0.4,
            "T": 0.2,
            "lambda_a": 0.05,
            "lambda_b": 0.2,
            "lambda_s": 1.0,
            "warmup_epoch": 60,
            "test_fusion_weight": 0.6,               
        }
    }

    run_loso_rskp_opportunity(config, device=device)
