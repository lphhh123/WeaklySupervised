# pre_train_rwhar.py
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from OtherData.RWHAR.pre_train.dataset_rwhar import RWHARDataset_10s
from OtherData.Opportunity.pre_train.pre_model_opportunity import CNN1DClassifier  # 你如果CNN1DClassifier不依赖opportunity命名，也可直接搬出来


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_pretrain_model(
    model,
    train_loader,
    eval_loader,
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

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_state = copy.deepcopy(model.backbone.state_dict())
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_state, save_path)
            print(f'>>> Best Eval Loss = {best_eval_loss:.4f}, saved backbone -> {save_path}')

        scheduler.step()

    print(f'[Done] best Eval Loss = {best_eval_loss:.4f}  | saved: {save_path}')


def run_loso_pretrain(config: dict):
    set_seed(config.get("seed", 2024))

    dataset_dir = config["dataset_dir"]
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    val_mode = config.get("val_mode", "test")

    for fold in range(config.get("num_folds", 15)):  # ★15 folds
        loso_json = f"loso_sbj_{fold}.json"
        print("\n" + "=" * 80)
        print(f"[LOSO] Fold {fold} | json={loso_json} | val_mode={val_mode}")
        print("=" * 80)

        train_ds_full = RWHARDataset_10s(
            dataset_dir=dataset_dir,
            loso_json=loso_json,
            split="train",
            fps=config.get("fps", 50),
            num_sensors=config.get("num_sensors", 21),
            win_sec=config.get("win_sec", 10.0),
            win_overlap=config.get("win_overlap", 0.5),
            normalize=True,
            stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
            ignore_zeros_in_stats=config.get("ignore_zeros_in_stats", False),
            cache_raw=config.get("cache_raw", False),
        )

        test_ds = RWHARDataset_10s(
            dataset_dir=dataset_dir,
            loso_json=loso_json,
            split="test",
            fps=config.get("fps", 50),
            num_sensors=config.get("num_sensors", 21),
            win_sec=config.get("win_sec", 10.0),
            win_overlap=config.get("win_overlap", 0.5),
            normalize=True,
            stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
            ignore_zeros_in_stats=config.get("ignore_zeros_in_stats", False),
            cache_raw=config.get("cache_raw", False),
        )

        eval_ds = test_ds
        train_ds = train_ds_full

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

        print(f"[Fold {fold}] train windows={len(train_ds)} | eval windows={len(eval_ds)} | test windows={len(test_ds)}")

        in_channels = config.get("in_channels", config.get("num_sensors", 21))
        model = CNN1DClassifier(
            num_classes=config["num_classes"],   # ★8
            task=config.get("task", "single"),
            feat_dim=config.get("feat_dim", 512),
            in_channels=in_channels,             # ★21
        )

        model_name = config.get("model_name", "CNN1D")
        save_backbone_path = os.path.join(out_dir, f"rwhar_{model_name}_pretrained_loso_sbj_{fold}.pth")

        train_pretrain_model(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_epochs=config.get("num_epochs", 60),
            lr=config.get("lr", 1e-3),
            save_path=save_backbone_path,
        )


if __name__ == "__main__":
    config = {
        "dataset_dir": "/home/lipei/TAL_data/rwhar/",
        "out_dir": "/home/lipei/project/WSDDN/RWHAR/pre_train",

        "num_folds": 15,
        "num_classes": 8,
        "task": "single",
        "model_name": "CNN1D",

        # RWHAR 数据参数
        "fps": 50,
        "num_sensors": 21,
        "in_channels": 21,


        "win_sec": 10.0,
        "win_overlap": 0.5,    # 10s窗口步长=5s（也可以改成0.0 -> 步长10s）

        # 归一化 stats json
        "stats_dirname": "loso_norm_stats_json",
        "ignore_zeros_in_stats": False,

        # 训练参数
        "batch_size": 32,
        "num_epochs": 60,
        "lr": 1e-3,
        "num_workers": 4,
        "seed": 2024,
        "feat_dim": 512,

        # RWHAR 序列长：建议 False，避免一次性缓存所有 raw
        "cache_raw": False,

        "val_mode": "test",
    }

    run_loso_pretrain(config)
