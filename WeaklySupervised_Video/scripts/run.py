import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.cola import build_model as build_cola_model
from models.rskp import build_model as build_rskp_model
from utils.config import load_config
from utils.seed import set_seed


DATASET_TO_CONFIG = {
    "xrfv2": "configs/xrfv2.json",
    "hangtime": "configs/hangtime.json",
    "opportunity": "configs/opportunity.json",
    "rwhar": "configs/rwhar.json",
    "sbhar": "configs/sbhar.json",
    "wear": "configs/wear.json",
    "wetlab": "configs/wetlab.json",
}


def build_cola_cfg(cfg: dict) -> SimpleNamespace:
    model_cfg = cfg.get("model", {})
    cola_cfg = model_cfg.get("cola", {})
    dataset_cfg = cfg.get("dataset", {})
    return SimpleNamespace(
        FEATS_DIM=model_cfg.get("inp_feat_num", 36),
        NUM_CLASSES=model_cfg.get("num_classes", 30),
        BACKBONE_TYPE=cola_cfg.get("backbone_type", "cnn1d"),
        PRETRAINED_PATH=cola_cfg.get("pretrained_path"),
        TRAIN_BACKBONE=cola_cfg.get("train_backbone", False),
        R_EASY=cola_cfg.get("r_easy", 15),
        R_HARD=cola_cfg.get("r_hard", 20),
        m=cola_cfg.get("m", 1),
        M=cola_cfg.get("M", 10),
        DATASET_NAME=dataset_cfg.get("name", "xrfv2"),
    )


def run_dry_run(cfg: dict, model_name: str, device: torch.device) -> None:
    dataset_cfg = cfg.get("dataset", {})
    num_segments = int(dataset_cfg.get("num_segments", 2048))
    num_segments = min(num_segments, int(dataset_cfg.get("dry_run_num_segments", 128)))
    batch_size = int(cfg.get("training", {}).get("batch_size", 2))

    if model_name == "cola":
        cola_cfg = build_cola_cfg(cfg)
        model = build_cola_model(cola_cfg).to(device)
        dummy = torch.randn(batch_size, num_segments, cola_cfg.FEATS_DIM, device=device)
        with torch.no_grad():
            video_scores, contrast_pairs, actionness, cas = model(dummy)
        print("[dry_run][cola] video_scores:", video_scores.shape)
        print("[dry_run][cola] actionness:", actionness.shape)
        print("[dry_run][cola] cas:", cas.shape)
        print("[dry_run][cola] contrast_pairs:", {k: v.shape for k, v in contrast_pairs.items()})
        return

    if model_name == "rskp":
        model = build_rskp_model(cfg).to(device)
        inp_feat_num = cfg.get("model", {}).get("inp_feat_num", 36)
        dummy = torch.randn(batch_size, num_segments, inp_feat_num, device=device)
        with torch.no_grad():
            out_a, out_b, aux = model(dummy)
        print("[dry_run][rskp] out_a lens:", [t.shape for t in out_a])
        print("[dry_run][rskp] out_b lens:", [t.shape for t in out_b])
        print("[dry_run][rskp] aux lens:", [t.shape for t in aux])
        return

    raise ValueError(f"Unknown model name: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified runner for CoLA/RSKP")
    parser.add_argument("--dataset", required=True, choices=DATASET_TO_CONFIG.keys())
    parser.add_argument("--model", required=True, choices=["cola", "rskp"])
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()

    config_path = args.config or DATASET_TO_CONFIG[args.dataset]
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(repo_root / config_path), repo_root=str(repo_root))
    cfg.setdefault("dataset", {})
    cfg["dataset"].setdefault("name", args.dataset)
    cfg.setdefault("model", {})
    cfg["model"].setdefault("type", args.model)

    set_seed(args.seed)
    device = torch.device(args.device)

    if args.dry_run:
        run_dry_run(cfg, args.model, device)
        return

    print(
        "[warn] Full training/testing pipeline is not wired in scripts/run.py yet. "
        "Running a minimal forward pass to validate configuration."
    )
    run_dry_run(cfg, args.model, device)
    return


if __name__ == "__main__":
    main()
