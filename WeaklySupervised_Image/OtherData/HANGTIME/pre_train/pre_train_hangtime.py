# pre_train_hangtime.py

import os
import copy
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from OtherData.HANGTIME.pre_train.dataset_hangtime import HangtimeDataset_3s
from pre_train.pre_model import CNN1DClassifier, VGG1DClassifier


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
     LOSO：
      - train_ds = split="train" (Training subjects)
      - test_ds  = split="test"  (Validation subject)
    default：eval_loader  test_ds（“trainingtest”）

    ：val_mode="train_subject" ， Training subjects  subject  1  val，
         test_ds test。
    """
    set_seed(config.get("seed", 2024))

    dataset_dir = config["dataset_dir"]
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    val_mode = config.get("val_mode", "test")

    for fold in range(config.get("loso_number", 1)):
        loso_json = f"loso_sbj_{fold}.json"
        print("\n" + "=" * 80)
        print(f"[LOSO] Fold {fold} | json={loso_json} | val_mode={val_mode}")
        print("=" * 80)

        # -------- 1) Train dataset：Training subjects --------
        train_ds_full = HangtimeDataset_3s(
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

        test_ds = HangtimeDataset_3s(
            dataset_dir=dataset_dir,
            loso_json=loso_json,
            split="test",
            fps=config.get("fps", 30),
            num_sensors=config.get("num_sensors", 113),
            win_sec=config.get("win_sec", 3.0),
            win_overlap=config.get("win_overlap", 0.5),
            normalize=True,
            stats_dirname=config.get("stats_dirname", "loso_norm_stats_json"),
            ignore_zeros_in_stats=config.get("ignore_zeros_in_stats", False),
        )

        eval_ds = test_ds

        if val_mode == "train_subject":
            train_subjects = list(getattr(train_ds_full, "subjects", []))
            if len(train_subjects) < 2:
                raise RuntimeError(f"[Fold {fold}] Not enough training subjects for val_mode=train_subject")

            rng = random.Random(config.get("seed", 2024) + fold)
            val_sbj = rng.choice(train_subjects)
            tr_subjects = [s for s in train_subjects if s != val_sbj]

            train_ds = HangtimeDataset_3s(
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
            val_ds = HangtimeDataset_3s(
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
            train_ds = train_ds_full

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
        model_name = str(config.get("model_name", "CNN1D"))

        if model_name == "CNN1D":
            model = CNN1DClassifier(
                num_classes=config["num_classes"],
                task=config.get("task", "single"),
                feat_dim=config.get("feat_dim", 512),
                in_channels=in_channels,
            )
        elif model_name == "VGG1D":
            model = VGG1DClassifier(
                num_classes=config["num_classes"],
                task=config.get("task", "single"),
                feat_dim=config.get("feat_dim", 512),
                in_channels=in_channels,
            )
        elif model_name == "VGG1D_BUAA":
            model = VGG1DClassifier(
                num_classes=config["num_classes"],
                task=config.get("task", "single"),
                feat_dim=config.get("feat_dim", 512),
                in_channels=in_channels,
            )
        else:
            raise ValueError(f"Unknown model_name={model_name}, choose from: CNN1D / VGG1D / VGG1D_BUAA")

        model_name = config.get("model_name", "CNN1D")
        save_backbone_path = os.path.join(out_dir, f"hangtime_{model_name}_pretrained_loso_sbj_{fold}.pth")

        # -------- 6) Train (save best by eval loss) --------
        train_pretrain_model(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            num_epochs=config.get("num_epochs", 60),
            lr=config.get("lr", 1e-3),
            save_path=save_backbone_path,
        )

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
        "dataset_dir": "/home/lipei/TAL_data/hangtime/",
        "out_dir": "/home/lipei/project/WSDDN/OtherData/HANGTIME/pre_train",

        "num_classes": 5,
        "task": "single",
        "model_name": "CNN1D",
        "fps": 50,
        "num_sensors": 3,
        "in_channels": 3,
        "loso_number":24,

        # 3s window
        "win_sec": 3.0,
        "win_overlap": 0.5,

        "stats_dirname": "loso_norm_stats_json",
        "ignore_zeros_in_stats": False,

        "batch_size": 32,
        "num_epochs": 60,
        "lr": 1e-3,
        "num_workers": 4,
        "seed": 2024,
        "feat_dim": 512,

        "val_mode": "test",
    }

    run_loso_pretrain(config)
