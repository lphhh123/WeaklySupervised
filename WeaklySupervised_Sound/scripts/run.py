import argparse
import importlib
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from models.dcase import CRNN
from models.cdur import CDur
from utils.config import load_json_config, merge_cli_overrides
from utils.dry_run import DummySpec, make_dummy_loader
from utils.losses import BCELossWithLabelSmoothing, RobustBCELoss
from utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified runner for weakly supervised audio/sequence localization")
    parser.add_argument("--dataset", required=True, choices=[
        "xrfv2", "hangtime", "opportunity", "rwhar", "sbhar", "wear", "wetlab"
    ])
    parser.add_argument("--model", required=True, choices=["dcase", "cdur"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--override", action="append", default=[], help="Override config key=value")
    return parser.parse_args()


def resolve_device(arg: str | None) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name: str, config: dict, in_channels: int, num_classes: int) -> torch.nn.Module:
    if model_name == "dcase":
        params = config["model"]["dcase"]["params"]
        return CRNN(n_in_channel=1, nclass=num_classes, **params)
    params = config["model"]["cdur"]["params"]
    return CDur(inputdim=in_channels, outputdim=num_classes, **params)


def build_dataset(name: str, split: str, config: dict):
    data_cfg = config["data"]
    paths = config["paths"]
    if name == "xrfv2":
        if split == "train":
            module = importlib.import_module("dataset.dataset_xrfv2")
            cls = getattr(module, "WeaklySupervisedXRFV2DatasetTrain")
            return cls(
                dataset_dir=paths["data_dir"],
                mapping_path=paths["mapping_path"],
                split="train",
                use_airpods=data_cfg.get("use_airpods", True),
            )
        module = importlib.import_module("dataset.dataset_xrfv2")
        cls = getattr(module, "WeaklySupervisedXRFV2DatasetTest")
        return cls(
            config={
                "path": {
                    "test_dataset_path": paths["data_dir"],
                    "dataset_root_path": paths["data_dir"],
                    "mapping_path": paths["mapping_path"],
                },
                "testing": config.get("testing", {}),
                "training": {"num_classes": data_cfg["num_classes"]},
            },
            use_airpods=data_cfg.get("use_airpods", True),
        )

    dataset_map = {
        "wear": ("OtherData.WEAR.dataset_wear_ws", "WeaklyWearDataset"),
        "hangtime": ("OtherData.HANGTIME.dataset_hangtime_ws", "WeaklyHangtimeDataset"),
        "opportunity": ("OtherData.Opportunity.dataset_opportunity_ws", "WeaklyOpportunityDataset"),
        "rwhar": ("OtherData.RWHAR.dataset_rwhar_ws", "WeaklyRWHARDataset"),
        "sbhar": ("OtherData.SBHAR.dataset_sbhar_ws", "WeaklySBHARDataset"),
        "wetlab": ("OtherData.WETLAB.dataset_wetlab_ws", "WeaklyWetlabDataset"),
    }
    module_path, class_name = dataset_map[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    mode = "train" if split == "train" else "test_window"
    return cls(
        dataset_dir=paths["data_dir"],
        loso_json=data_cfg.get("loso_json", "loso_sbj_0.json"),
        mode=mode,
        fps=data_cfg["fps"],
        num_sensors=data_cfg["in_channels"],
        clip_sec=data_cfg["clip_sec"],
        num_classes=data_cfg["num_classes"],
        stats_dirname=data_cfg.get("stats_dirname", "loso_norm_stats_json"),
    )


def build_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)):
        x = batch[0]
        y = batch[1]
        return x, y
    raise ValueError("Unexpected batch format")


def run_train_step(model, batch, device, model_name: str, criterion):
    x, y = unpack_batch(batch)
    x = x.to(device)
    y = y.to(device)
    if model_name == "dcase":
        _, weak = model(x)
        loss = criterion(weak, y)
        return loss, weak
    x = x.transpose(1, 2)
    decision, _ = model(x)
    loss = criterion(decision, y)
    return loss, decision


def run_eval_step(model, batch, device, model_name: str):
    x, _ = unpack_batch(batch)
    x = x.to(device)
    if model_name == "dcase":
        _, weak = model(x)
        return weak
    x = x.transpose(1, 2)
    decision, _ = model(x)
    return decision


def run_for_dataset(args: argparse.Namespace, config: dict):
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_cfg = config["data"]
    train_cfg = config["training"]

    model = build_model(args.model, config, data_cfg["in_channels"], data_cfg["num_classes"]).to(device)

    if args.model == "dcase":
        criterion = BCELossWithLabelSmoothing(label_smoothing=train_cfg.get("label_smoothing", 0.0))
    else:
        criterion = RobustBCELoss(label_smoothing=train_cfg.get("label_smoothing", 0.0))

    if args.dry_run:
        spec = DummySpec(
            num_samples=4,
            in_channels=data_cfg["in_channels"],
            clip_frames=data_cfg["clip_frames"],
            num_classes=data_cfg["num_classes"],
            seed=args.seed,
        )
        loader = make_dummy_loader(spec, batch_size=train_cfg.get("batch_size", 2))
        model.train()
        for step, batch in enumerate(loader):
            loss, _ = run_train_step(model, batch, device, args.model, criterion)
            loss.backward()
            if step >= 1:
                break
        print("[dry_run] Completed forward/backward passes.")
        return

    if args.mode == "train":
        dataset = build_dataset(args.dataset, "train", config)
        loader = build_loader(dataset, train_cfg.get("batch_size", 2), train_cfg.get("num_workers", 0))
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-4))
        model.train()
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            loss, _ = run_train_step(model, batch, device, args.model, criterion)
            loss.backward()
            optimizer.step()
            if step >= train_cfg.get("max_steps", 2):
                break
        print("Train loop finished.")
        return

    dataset = build_dataset(args.dataset, "test", config)
    model.eval()
    if hasattr(dataset, "dataset") and not hasattr(dataset, "__len__"):
        with torch.no_grad():
            for file_name, data_iter in dataset.dataset():
                for idx, (clip_dict, _) in enumerate(data_iter):
                    x = clip_dict["imu"].unsqueeze(0).to(device)
                    if args.model == "dcase":
                        model(x)
                    else:
                        model(x.transpose(1, 2))
                    if idx >= 1:
                        break
                break
        print("Test loop finished (streaming dataset).")
        return

    loader = build_loader(dataset, train_cfg.get("batch_size", 2), train_cfg.get("num_workers", 0))
    with torch.no_grad():
        for step, batch in enumerate(loader):
            run_eval_step(model, batch, device, args.model)
            if step >= 1:
                break
    print("Test loop finished.")


def main():
    args = parse_args()
    config_path = args.config
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / f"{args.dataset}.json"
    config = load_json_config(config_path)
    if args.override:
        config = merge_cli_overrides(config, args.override)
    run_for_dataset(args, config)


if __name__ == "__main__":
    main()
