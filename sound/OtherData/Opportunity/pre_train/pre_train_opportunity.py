# pre_train_opportunity.py

import os
import copy
import random
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from OtherData.Opportunity.pre_train.dataset_opportunity import OpportunityDataset_3s
from pre_train.pre_model import CNN1DClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_pretrain_model(
    model,
    train_loader,
    eval_loader,             # 这里 eval_loader 可以是 test 或 val
    num_epochs=50,
    lr=1e-3,
    save_path='CNN1D_best_backbone.pth',
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() if model.task == 'single' else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        # -------- 1) Train --------
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0

        for data, labels in tqdm(train_loader, desc=f'[Train] Epoch {epoch + 1}/{num_epochs}'):
            data = data.to(device)
            labels = labels.to(device)
            if model.task == 'single':
                labels = labels.long()

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * data.size(0)
            if model.task == 'single':
                preds = outputs.argmax(dim=1)
                total_train_acc += (preds == labels).sum().item()
            else:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_train_acc += (preds == labels).sum().item() / labels.size(1)

        avg_train_loss = total_train_loss / max(1, len(train_loader.dataset))
        avg_train_acc = total_train_acc / max(1, len(train_loader.dataset))

        # -------- 2) Eval (val 或 test) --------
        model.eval()
        total_eval_loss = 0.0
        total_eval_acc = 0.0

        with torch.no_grad():
            for data, labels in tqdm(eval_loader, desc=f'[Eval ] Epoch {epoch + 1}/{num_epochs}'):
                data = data.to(device)
                labels = labels.to(device)
                if model.task == 'single':
                    labels = labels.long()

                outputs = model(data)
                loss = criterion(outputs, labels)

                total_eval_loss += loss.item() * data.size(0)

                if model.task == 'single':
                    preds = outputs.argmax(dim=1)
                    total_eval_acc += (preds == labels).sum().item()
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    total_eval_acc += (preds == labels).sum().item() / labels.size(1)

        avg_eval_loss = total_eval_loss / max(1, len(eval_loader.dataset))
        avg_eval_acc = total_eval_acc / max(1, len(eval_loader.dataset))

        print(
            f'Epoch {epoch + 1}/{num_epochs} | '
            f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | '
            f'Eval  Loss: {avg_eval_loss:.4f}, Eval  Acc: {avg_eval_acc:.4f}'
        )

        # -------- 3) Save best backbone by eval_loss --------
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_state = copy.deepcopy(model.backbone.state_dict())
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_state, save_path)
            print(f'>>> Best Eval Loss = {best_eval_loss:.4f}, saved backbone -> {save_path}')

        scheduler.step()

    print(f'[Done] best Eval Loss = {best_eval_loss:.4f}  | saved: {save_path}')


def run_loso_pretrain(config: dict):
    """
    四折 LOSO：
      - train_ds = split="train" (Training subjects)
      - test_ds  = split="test"  (Validation subject)
    默认：eval_loader 直接用 test_ds（即“明确训练和测试划分”）

    可选：val_mode="train_subject" 时，从 Training subjects 中按 subject 抽 1 个做 val，
         test_ds 仍然保留用于最终测试。
    """
    set_seed(config.get("seed", 2024))

    dataset_dir = config["dataset_dir"]
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    val_mode = config.get("val_mode", "test")  # "test" 或 "train_subject"

    for fold in range(config.get("loso_number", 1)):
        loso_json = f"loso_sbj_{fold}.json"
        print("\n" + "=" * 80)
        print(f"[LOSO] Fold {fold} | json={loso_json} | val_mode={val_mode}")
        print("=" * 80)

        # -------- 1) Train dataset：Training subjects --------
        train_ds_full = OpportunityDataset_3s(
            dataset_dir=dataset_dir,
            loso_json=loso_json,
            split="train",
            fps=config.get("fps", 30),
            num_sensors=config.get("num_sensors", 113),
            win_sec=config.get("win_sec", 3.0),
            win_overlap=config.get("win_overlap", 0.5),
            normalize=True,
            stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
            ignore_zeros_in_stats=config.get("ignore_zeros_in_stats", False),
        )

        # -------- 2) Test dataset：Validation subject（LOSO规定的测试人）--------
        test_ds = OpportunityDataset_3s(
            dataset_dir=dataset_dir,
            loso_json=loso_json,
            split="test",
            fps=config.get("fps", 30),
            num_sensors=config.get("num_sensors", 113),
            win_sec=config.get("win_sec", 3.0),
            win_overlap=config.get("win_overlap", 0.5),
            normalize=True,  # 使用同一 fold 的 mean/var
            stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
            ignore_zeros_in_stats=config.get("ignore_zeros_in_stats", False),
        )

        # -------- 3) 选择 eval 数据集 --------
        # 默认：eval 就用 test（要的“明确训练/测试划分”）
        eval_ds = test_ds

        # （不用测试集挑最优）——从 Training subjects 里抽一个 subject 做 val
        if val_mode == "train_subject":
            # 取 train_ds_full.subjects 列表（Dataset 内部已有 subjects）
            train_subjects = list(getattr(train_ds_full, "subjects", []))
            if len(train_subjects) < 2:
                raise RuntimeError(f"[Fold {fold}] Not enough training subjects for val_mode=train_subject")

            rng = random.Random(config.get("seed", 2024) + fold)
            val_sbj = rng.choice(train_subjects)
            tr_subjects = [s for s in train_subjects if s != val_sbj]

            # 这里仍用 fold 的统计
            train_ds = OpportunityDataset_3s(
                dataset_dir=dataset_dir,
                loso_json=loso_json,
                split="train",
                subjects=tr_subjects,
                fps=config.get("fps", 30),
                num_sensors=config.get("num_sensors", 113),
                win_sec=config.get("win_sec", 3.0),
                win_overlap=config.get("win_overlap", 0.5),
                normalize=True,
                stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
                ignore_zeros_in_stats=config.get("ignore_zeros_in_stats", False),
            )
            val_ds = OpportunityDataset_3s(
                dataset_dir=dataset_dir,
                loso_json=loso_json,
                split="train",
                subjects=[val_sbj],
                fps=config.get("fps", 30),
                num_sensors=config.get("num_sensors", 113),
                win_sec=config.get("win_sec", 3.0),
                win_overlap=config.get("win_overlap", 0.5),
                normalize=True,
                stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
                ignore_zeros_in_stats=config.get("ignore_zeros_in_stats", False),
            )
            eval_ds = val_ds
        else:
            train_ds = train_ds_full  # 直接用 Training subjects 的所有窗口训练

        # -------- 4) Dataloaders --------
        train_loader = DataLoader(
            train_ds,
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )

        print(f"[Fold {fold}] train windows={len(train_ds)} | eval windows={len(eval_ds)} | test windows={len(test_ds)}")

        # -------- 5) Build model --------
        in_channels = config.get("in_channels", config.get("num_sensors", 113))
        model = CNN1DClassifier(
            num_classes=config["num_classes"],
            task=config.get("task", "single"),
            feat_dim=config.get("feat_dim", 512),
            in_channels=in_channels,
        )

        model_name = config.get("model_name", "CNN1D")
        save_backbone_path = os.path.join(out_dir, f"opportunity_{model_name}_pretrained_loso_sbj_{fold}.pth")

        # -------- 6) Train (save best by eval loss) --------
        train_pretrain_model(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_epochs=config.get("num_epochs", 60),
            lr=config.get("lr", 1e-3),
            save_path=save_backbone_path,
        )

        # -------- 7) 训练结束后，在 test 上跑一遍最终指标 --------
        # 如果 val_mode="test"，eval 就是 test，这一步可省略
        if val_mode == "train_subject":
            print(f"[Fold {fold}] Final evaluation on LOSO test subject ...")
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            criterion = nn.CrossEntropyLoss()

            total_loss, total_acc = 0.0, 0.0
            with torch.no_grad():
                for data, labels in tqdm(test_loader, desc=f"[Test] Fold {fold}"):
                    data = data.to(device)
                    labels = labels.to(device).long()
                    logits = model(data)
                    loss = criterion(logits, labels)
                    total_loss += loss.item() * data.size(0)
                    total_acc += (logits.argmax(1) == labels).sum().item()
            avg_loss = total_loss / max(1, len(test_loader.dataset))
            avg_acc = total_acc / max(1, len(test_loader.dataset))
            print(f"[Test] Fold {fold} | Loss={avg_loss:.4f} | Acc={avg_acc:.4f}")


if __name__ == "__main__":
    config = {
        "dataset_dir": "/home/lipei/TAL_data/opportunity/",
        "out_dir": "/home/lipei/project/WSDDN/OtherData/Opportunity/pre_train",

        "num_classes": 17,
        "task": "single",
        "model_name": "CNN1D",

        # 数据参数
        "fps": 30,
        "num_sensors": 113,
        "in_channels": 113,
        "loso_number":4,

        # 3s window
        "win_sec": 3.0,
        "win_overlap": 0.5,

        # 归一化 stats 的位置（已经算好 mean/var 的 json）
        "stats_dirname": "loso_norm_stats_json",
        "ignore_zeros_in_stats": False,

        # 训练参数
        "batch_size": 32,
        "num_epochs": 60,
        "lr": 1e-3,
        "num_workers": 4,
        "seed": 2024,
        "feat_dim": 512,

        # 关键：验证模式
        # "test"：eval=LOSO测试人（完全按“明确训练/测试划分”）
        # "train_subject"：从Training subjects里抽1个人做val，test仍用于最终评估（更严谨）
        "val_mode": "test",
    }

    run_loso_pretrain(config)