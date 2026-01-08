# train_epoch.py 里

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTrain
from builder_models import build_wsddn_imu_model, build_pcl_oicr_imu_model
from tool import load_label_mapping
from builder_pretrainbackbone import load_pretrained_backbone, get_pretrained_spec


def train_wsddn_imu(config, exp_name="wsddn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备：{device}")
    print(f"实验名：{exp_name}")

    # === 统一读取 use_airpods 开关 ===
    use_airpods = bool(config["training"].get("use_airpods", False))
    in_channels = 30 + (6 if use_airpods else 0)  # IMU:30, AirPods:6 → 36
    print(f"use_airpods = {use_airpods}, in_channels = {in_channels}")

    # 判断预训练ckpt是否存在，用于决定是否强制训练backbone ===
    spec = get_pretrained_spec(config["model"]["pretrained_name"])
    pretrain_ckpt = spec.get("ckpt")
    backbone_loaded = (pretrain_ckpt is not None and os.path.exists(pretrain_ckpt))  #True:使用预训练模型；False：后续将随机初始化预训练模型并参与训练
    print(f"[Train] pretrained_name={config['model']['pretrained_name']}")
    print(f"[Train] backbone_ckpt={pretrain_ckpt}")
    print(f"[Train] backbone_loaded={backbone_loaded}")

    # 1. 加载backbone（加上 in_channels）
    pretrained_backbone, T_global = load_pretrained_backbone(
        pretrained_name=config["model"]["pretrained_name"],
        device=device,
        in_channels=in_channels,
    )

    pretrained_backbone = pretrained_backbone.to(device)

    # 2. 加载标签映射
    _, _, new_to_action = load_label_mapping(config["path"]["mapping_path"])
    num_classes = len(new_to_action)
    print(f"类别数：{num_classes}")

    # 3. 初始化模型
    model = build_wsddn_imu_model(config, num_classes, device)

    # 4. 加载训练数据集
    train_dataset = WeaklySupervisedXRFV2DatasetTrain(
        dataset_dir=config["path"]["train_dataset_path"],
        mapping_path=config["path"]["mapping_path"],
        split="train",
        num_proposals=config["training"]["num_proposals"],
        T_global=T_global,
        use_airpods=use_airpods,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )
    print(f"训练集样本数：{len(train_dataset)}，批次大小：{config['training']['batch_size']}")

    # backbone 是否参与训练（若预训练不存在 -> 强制参与训练） ===
    train_backbone = bool(config["training"].get("train_backbone", False))
    if not backbone_loaded:
        train_backbone = True
        print("[Train] 预训练不存在 -> 强制 train_backbone=True（随机初始化并参与训练）")
    print(f"train_backbone = {train_backbone}")

    if train_backbone:
        pretrained_backbone.train()
        for p in pretrained_backbone.parameters():
            p.requires_grad_(True)
    else:
        pretrained_backbone.eval()
        for p in pretrained_backbone.parameters():
            p.requires_grad_(False)

    # 5. 优化器和调度器
    backbone_lr = float(config["training"].get("backbone_lr", config["training"]["lr"] * 0.1))
    param_groups = [{"params": model.parameters(), "lr": config["training"]["lr"]}]
    if train_backbone:
        param_groups.append({"params": pretrained_backbone.parameters(), "lr": backbone_lr})
    optimizer = optim.Adam(param_groups, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["lr_step_size"],
        gamma=config["training"]["lr_gamma"]
    )

    # 6. 损失函数
    criterion = nn.BCELoss()

    # 7. 训练过程
    best_loss = float('inf')
    ckpt_dir = config["path"]["checkpoint_path"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{exp_name}.pth")
    # 日志文件：记录每个 epoch 的损失和学习率
    log_path = config["path"]["result_path"]
    train_log_path = os.path.join(log_path, f"{exp_name}_train_log.csv")
    # 如果文件不存在，写入表头
    if not os.path.exists(train_log_path):
        with open(train_log_path, "w") as f:
            f.write("epoch,avg_loss,best_loss,lr\n")


    lambda_reg = float(config["training"].get("spatial_reg_weight", 0.0))
    iou_thresh = float(config["training"].get("spatial_reg_iou", 0.8))
    print(f"空间正则权重 λ = {lambda_reg}, IoU 阈值 = {iou_thresh}")

    num_epochs = config["training"]["num_epochs"]

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (sample_30s, proposal_boxes, labels) in enumerate(progress_bar):
            # 数据移至设备
            sample_30s = sample_30s.to(device)
            proposal_boxes = proposal_boxes.to(device)
            labels = labels.to(device).float()
            B = sample_30s.shape[0]

            # 冻结backbone时用 no_grad；参与训练时不用 ===
            if train_backbone:
                global_feat = pretrained_backbone(sample_30s)
            else:
                with torch.no_grad():
                    global_feat = pretrained_backbone(sample_30s)

            outputs = model(global_feat, proposal_boxes)

            video_prob = outputs["video_prob"]   # [B, C]
            joint_prob = outputs["joint_prob"]   # [B, P, C]
            feat_fc7 = outputs["feat_fc7"]       # [B, P, D]  —— 请确认 WSDDN_model.py 里有返回

            # ==== 先做数值检查 ====
            if torch.isnan(video_prob).any():
                print(f"[NaN DETECTED] epoch={epoch}, batch={batch_idx}")
                print("  video_prob min:", video_prob.min().item(), "max:", video_prob.max().item())
                raise ValueError("video_prob contains NaN")

            # ==== clamp 到 (0,1) 内，避免 BCE 的 assert  ====
            video_prob_safe = torch.clamp(video_prob, min=1e-6, max=1 - 1e-6)

            # -------- 1) 主损失：多标签 BCE --------
            bce_loss = criterion(video_prob_safe, labels)

            # -------- 2) 空间正则项 --------
            spatial_reg = torch.tensor(0.0, device=device)
            reg_count = 0

            if lambda_reg > 0.0:
                # 遍历 batch 内每个样本
                for b in range(B):
                    # 这个样本有哪些“正类”
                    pos_cls = (labels[b] > 0).nonzero(as_tuple=False).view(-1)  # [N_pos]
                    if pos_cls.numel() == 0:
                        continue

                    # 当前样本的 proposal boxes 和 fc7 特征
                    boxes_b = proposal_boxes[b].float()  # [P, 2]
                    feats_b = feat_fc7[b]               # [P, D]

                    for cls_idx in pos_cls:
                        cls_idx = cls_idx.item()

                        # 2.1 找这个类别下得分最高的 proposal
                        scores_bc = joint_prob[b, :, cls_idx]  # [P]
                        p_star = int(scores_bc.argmax().item())  # index of max-scoring proposal

                        box_star = boxes_b[p_star]  # [2]
                        start_star, end_star = box_star[0], box_star[1]

                        # 2.2 计算其他 proposal 与它的 IoU（1D 时间段）
                        start_all = boxes_b[:, 0]  # [P]
                        end_all = boxes_b[:, 1]    # [P]

                        tt1 = torch.maximum(start_all, start_star)
                        tt2 = torch.minimum(end_all, end_star)
                        inter = (tt2 - tt1).clamp(min=0)

                        len_all = (end_all - start_all).clamp(min=1e-6)
                        len_star = (end_star - start_star).clamp(min=1e-6)
                        union = len_all + len_star - inter
                        iou = inter / union  # [P]

                        # 2.3 选出 IoU 大于阈值、且不是自己本身的那些 proposal
                        idx_all = torch.arange(boxes_b.size(0), device=device)
                        neighbor_mask = (iou > iou_thresh) & (idx_all != p_star)

                        if neighbor_mask.any():
                            f_star = feats_b[p_star].unsqueeze(0)  # [1, D]
                            f_nb = feats_b[neighbor_mask]          # [K, D]

                            # L2 距离：希望邻居特征接近最高分特征
                            diff = f_nb - f_star  # [K, D]
                            reg_term = diff.pow(2).mean()  # 一个标量

                            spatial_reg = spatial_reg + reg_term
                            reg_count += 1

            # 2.4 合并损失
            if lambda_reg > 0.0 and reg_count > 0:
                spatial_reg = spatial_reg / reg_count
                total_loss = bce_loss + lambda_reg * spatial_reg
            else:
                total_loss = bce_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * B
            progress_bar.set_postfix({
                "batch_loss": f"{total_loss.item():.4f}",
                "bce": f"{bce_loss.item():.4f}",
                "reg": f"{float(spatial_reg.item()) if reg_count>0 else 0.0:.4f}"
            })

        # 计算平均损失
        avg_loss = epoch_loss / len(train_dataset)
        current_lr = scheduler.get_last_lr()[0]

        # 保存ckpt时，如果backbone参与训练 -> 必须保存 backbone_state_dict ===
        if avg_loss < best_loss:
            best_loss = avg_loss

            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "epoch": epoch + 1,

                # test用来构建一致的backbone
                "pretrained_name": config["model"].get("pretrained_name", None),
                "backbone_loaded": backbone_loaded,
                "train_backbone": train_backbone,
                "in_channels": in_channels,
                "use_airpods": use_airpods,
            }

            # 只要参与训练（包括：预训练微调 / 随机初始化训练），就把backbone权重存进去
            if train_backbone:
                save_dict["backbone_state_dict"] = pretrained_backbone.state_dict()

            torch.save(save_dict, ckpt_path)
            print(f"保存最优模型至：{ckpt_path}（损失：{best_loss:.4f}）")

        # === 将训练损失记录到文件 ===
        with open(train_log_path, "a") as f:
            f.write(f"{epoch + 1},{avg_loss:.6f},{best_loss:.6f},{current_lr:.6f}\n")

        print(f"Epoch {epoch + 1} | 平均损失：{avg_loss:.4f} | 学习率：{current_lr:.6f}")

        # 学习率调度（放在最后一步）
        scheduler.step()

    print("训练完成！")
    return ckpt_path  # 返回最优模型路径

def train_pcl_imu(config, exp_name="pcl_imu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PCL/OICR] 训练设备：{device}")
    print(f"[PCL/OICR] 实验名：{exp_name}")

    # ★ 统一读取 use_airpods 开关
    use_airpods = bool(config["training"].get("use_airpods", False))
    in_channels = 30 + (6 if use_airpods else 0)
    print(f"[PCL/OICR] use_airpods = {use_airpods}, in_channels = {in_channels}")

    # 先判断预训练ckpt是否存在，用于决定是否强制训练backbone ===
    spec = get_pretrained_spec(config["model"]["pretrained_name"])
    pretrain_ckpt = spec.get("ckpt")
    backbone_loaded = (pretrain_ckpt is not None and os.path.exists(pretrain_ckpt))
    print(f"[PCL/OICR-Train] pretrained_name={config['model']['pretrained_name']}")
    print(f"[PCL/OICR-Train] backbone_ckpt={pretrain_ckpt}")
    print(f"[PCL/OICR-Train] backbone_loaded={backbone_loaded}")

    # 加载 backbone（存在则加载预训练；不存在则随机初始化）
    pretrained_backbone, T_global = load_pretrained_backbone(
        pretrained_name=config["model"]["pretrained_name"],
        device=device,
        in_channels=in_channels,
    )
    pretrained_backbone = pretrained_backbone.to(device)

    # 2. 类别数
    _, _, new_to_action = load_label_mapping(config["path"]["mapping_path"])
    num_classes = len(new_to_action)
    print(f"[PCL/OICR] num_classes = {num_classes}, T_global = {T_global}")

    # 3. 构建 PCL/OICR 模型
    model = build_pcl_oicr_imu_model(config, num_classes, device)

    # 4. 训练数据集
    train_dataset = WeaklySupervisedXRFV2DatasetTrain(
        dataset_dir=config["path"]["train_dataset_path"],
        mapping_path=config["path"]["mapping_path"],
        split="train",
        num_proposals=config["training"]["num_proposals"],
        T_global=T_global,
        use_airpods=use_airpods,   # ★
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )
    print(f"[PCL/OICR] 训练集样本数：{len(train_dataset)}")

    # backbone 是否参与训练（若预训练不存在 -> 强制参与训练）===
    train_backbone = bool(config["training"].get("train_backbone", False))
    if not backbone_loaded:
        train_backbone = True
        print("[PCL/OICR-Train] 预训练不存在 -> 强制 train_backbone=True（随机初始化并参与训练）")
    print(f"[PCL/OICR-Train] train_backbone = {train_backbone}")

    if train_backbone:
        pretrained_backbone.train()
        for p in pretrained_backbone.parameters():
            p.requires_grad_(True)
    else:
        pretrained_backbone.eval()
        for p in pretrained_backbone.parameters():
            p.requires_grad_(False)

    # 5. 优化器和调度器
    # param_groups，train_backbone=True 时把 backbone 参数加入 optimizer ===
    backbone_lr = float(config["training"].get("backbone_lr", config["training"]["lr"] * 0.1))
    param_groups = [{"params": model.parameters(), "lr": config["training"]["lr"]}]
    if train_backbone:
        param_groups.append({"params": pretrained_backbone.parameters(), "lr": backbone_lr})
    optimizer = optim.Adam(param_groups, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["lr_step_size"],
        gamma=config["training"]["lr_gamma"]
    )

    # 6. 训练循环
    num_epochs = config["training"]["num_epochs"]
    best_loss = float("inf")
    ckpt_dir = config["path"]["checkpoint_path"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{exp_name}.pth")

    log_path = config["path"]["result_path"]
    os.makedirs(log_path, exist_ok=True)
    train_log_path = os.path.join(log_path, f"{exp_name}_train_log.csv")
    if not os.path.exists(train_log_path):
        with open(train_log_path, "w") as f:
            f.write("epoch,avg_loss,best_loss,lr\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for sample_30s, proposal_boxes, labels in tqdm(
            train_loader, desc=f"[PCL/OICR] Epoch {epoch+1}/{num_epochs}"
        ):
            sample_30s = sample_30s.to(device)          # [B, 30 or 36, 2048]
            proposal_boxes = proposal_boxes.to(device)  # [B, P, 2]
            labels = labels.to(device)                  # [B, C]

            # 冻结backbone时用 no_grad；参与训练时不用 ===
            if train_backbone:
                global_feat = pretrained_backbone(sample_30s)
            else:
                with torch.no_grad():
                    global_feat = pretrained_backbone(sample_30s)

            outputs = model(global_feat, proposal_boxes, labels)
            losses = outputs["losses"]
            total_loss = sum(losses.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * sample_30s.size(0)

        epoch_loss /= len(train_dataset)
        current_lr = scheduler.get_last_lr()[0]
        print(f"[PCL/OICR] Epoch {epoch+1} | 平均损失: {epoch_loss:.4f}")

        with open(train_log_path, "a") as f:
            f.write(f"{epoch + 1},{epoch_loss:.6f},{best_loss:.6f},{current_lr:.6f}\n")

            # 保存ckpt时，如果backbone参与训练 -> 必须保存 backbone_state_dict ===
            if epoch_loss < best_loss:
                best_loss = epoch_loss

                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "epoch": epoch + 1,
                    "num_classes": num_classes,

                    # 元信息：test要用
                    "pretrained_name": config["model"].get("pretrained_name", None),
                    "backbone_loaded": backbone_loaded,
                    "train_backbone": train_backbone,
                    "use_airpods": use_airpods,
                    "in_channels": in_channels,
                }

                # 只要参与训练（包括：预训练微调 / 随机初始化训练），就把backbone权重存进去
                if train_backbone:
                    save_dict["backbone_state_dict"] = pretrained_backbone.state_dict()

                torch.save(save_dict, ckpt_path)
                print(f"[PCL/OICR] -> New best, save to {ckpt_path}")

        scheduler.step()

    print(f"[PCL/OICR] 训练完成，best_loss={best_loss:.4f}")
    return ckpt_path

