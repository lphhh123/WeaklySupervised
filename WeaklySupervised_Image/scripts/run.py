#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from builder_models import build_wsddn_imu_model, build_pcl_oicr_imu_model
from scripts.config_utils import load_json_config


DATASET_CONFIGS = {
    "xrfv2": {
        "wsddn": "configs/xrfv2_wsddn.json",
        "pcl": "configs/xrfv2_pcl.json",
        "oicr": "configs/xrfv2_oicr.json",
    },
    "hangtime": {
        "wsddn": "configs/hangtime_wsddn.json",
        "pcl": "configs/hangtime_pcl.json",
        "oicr": "configs/hangtime_oicr.json",
    },
    "opportunity": {
        "wsddn": "configs/opportunity_wsddn.json",
        "pcl": "configs/opportunity_pcl.json",
        "oicr": "configs/opportunity_oicr.json",
    },
    "rwhar": {
        "wsddn": "configs/rwhar_wsddn.json",
        "pcl": "configs/rwhar_pcl.json",
        "oicr": "configs/rwhar_oicr.json",
    },
    "sbhar": {
        "wsddn": "configs/sbhar_wsddn.json",
        "pcl": "configs/sbhar_pcl.json",
        "oicr": "configs/sbhar_oicr.json",
    },
    "wear": {
        "wsddn": "configs/wear_wsddn.json",
        "pcl": "configs/wear_pcl.json",
        "oicr": "configs/wear_oicr.json",
    },
    "wetlab": {
        "wsddn": "configs/wetlab_wsddn.json",
        "pcl": "configs/wetlab_pcl.json",
        "oicr": "configs/wetlab_oicr.json",
    },
}


DRYRUN_CLASSES = {
    "xrfv2": 30,
    "hangtime": 5,
    "opportunity": 17,
    "rwhar": 8,
    "sbhar": 12,
    "wear": 18,
    "wetlab": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified runner for weakly-supervised IMU models")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--model", required=True, choices=["wsddn", "oicr", "pcl"])
    parser.add_argument("--config", default=None, help="Path to JSON config (optional)")
    parser.add_argument("--dry_run", action="store_true", help="Run dry-run with dummy data")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Force device")
    return parser.parse_args()


def build_dummy_batch(
    batch_size: int,
    num_proposals: int,
    num_classes: int,
    feat_dim: int,
    t_global: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_feat = torch.randn(batch_size, feat_dim, t_global)

    starts = torch.linspace(0, t_global - 2, steps=num_proposals).long()
    ends = torch.clamp(starts + 2, max=t_global - 1)
    proposal_boxes = torch.stack([starts, ends], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)

    labels = torch.zeros(batch_size, num_classes)
    labels[:, : max(1, num_classes // 3)] = 1.0
    return global_feat, proposal_boxes, labels


def run_dry_run(dataset: str, model: str, device: torch.device) -> None:
    num_classes = DRYRUN_CLASSES[dataset]
    feat_dim = 512
    t_global = 64
    num_proposals = 12
    batch_size = 2

    config = {
        "model": {
            "type": "wsddn" if model == "wsddn" else ("pcl_imu" if model == "pcl" else "oicr_imu"),
            "pretrained_name": "CNN1D",
            "wsddn": {
                "spp_levels": [1, 2, 4],
                "spp_pool": "avg",
            },
            "refine_times": 2,
            "fg_thresh": 0.5,
            "bg_thresh": 0.1,
            "use_pcl": model == "pcl",
        },
        "training": {
            "num_proposals": num_proposals,
            "batch_size": batch_size,
        },
    }

    if model == "wsddn":
        model_instance = build_wsddn_imu_model(config, num_classes=num_classes, device=device)
    else:
        model_instance = build_pcl_oicr_imu_model(config, num_classes=num_classes, device=device)

    model_instance.train()

    global_feat, proposal_boxes, labels = build_dummy_batch(
        batch_size=batch_size,
        num_proposals=num_proposals,
        num_classes=num_classes,
        feat_dim=feat_dim,
        t_global=t_global,
    )
    dataset_obj = TensorDataset(global_feat, proposal_boxes, labels)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)

    for step, (feat, boxes, lab) in enumerate(loader, start=1):
        feat = feat.to(device)
        boxes = boxes.to(device)
        lab = lab.to(device)

        if model == "wsddn":
            outputs = model_instance(feat, boxes)
            video_prob = outputs["video_prob"]
            loss = torch.nn.functional.binary_cross_entropy(
                torch.clamp(video_prob, min=1e-6, max=1 - 1e-6),
                lab,
            )
        else:
            outputs = model_instance(feat, boxes, labels=lab)
            loss = sum(outputs["losses"].values())

        print(f"[DryRun] step={step} loss={float(loss):.6f}")
        if step >= 2:
            break

    print("[DryRun] completed")


def run_real(dataset: str, model: str, config_path: str | None, device: torch.device | None) -> None:
    default_config = DATASET_CONFIGS[dataset][model]
    config, loaded_path = load_json_config(default_config, config_path)
    print(f"[Config] loaded {loaded_path}")

    if device is not None:
        config["device"] = str(device)

    if dataset == "xrfv2":
        from run_main_xrfv2 import train, test

        exp_name = config.get("exp_name", f"{dataset}_{model}")
        ckpt_path = train(config, exp_name=exp_name)
        test(config, ckpt_path, "test_full")
        test(config, ckpt_path, "test_window")
        return

    dataset_map = {
        "hangtime": {
            "wsddn": "OtherData.HANGTIME.run_wsddn_hangtime",
            "pcl": "OtherData.HANGTIME.run_pcl_hangtime",
            "oicr": "OtherData.HANGTIME.run_oicrBUAA_hangtime",
        },
        "opportunity": {
            "wsddn": "OtherData.Opportunity.run_wsddn_opportunity",
            "pcl": "OtherData.Opportunity.run_pcl_opportunity",
            "oicr": "OtherData.Opportunity.run_oicrBUAA_opportunity",
        },
        "rwhar": {
            "wsddn": "OtherData.RWHAR.run_wsddn_rwhar",
            "pcl": "OtherData.RWHAR.run_pcl_rwhar",
            "oicr": "OtherData.RWHAR.run_oicrBUAA_rwhar",
        },
        "sbhar": {
            "wsddn": "OtherData.SBHAR.run_wsddn_sbhar",
            "pcl": "OtherData.SBHAR.run_pcl_sbhar",
            "oicr": "OtherData.SBHAR.run_oicrBUAA_sbhar",
        },
        "wear": {
            "wsddn": "OtherData.WEAR.run_wsddn_wear",
            "pcl": "OtherData.WEAR.run_pcl_wear",
            "oicr": "OtherData.WEAR.run_oicrBUAA_wear",
        },
        "wetlab": {
            "wsddn": "OtherData.WETLAB.run_wsddn_wetlab",
            "pcl": "OtherData.WETLAB.run_pcl_wetlab",
            "oicr": "OtherData.WETLAB.run_oicrBUAA_wetlab",
        },
    }

    target = dataset_map[dataset][model]
    module = __import__(target, fromlist=["run_entry"])
    if hasattr(module, "run_entry"):
        module.run_entry(config)
    else:
        raise RuntimeError(f"{target} does not expose run_entry(config)")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else None

    if args.dry_run:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_dry_run(args.dataset, args.model, device)
        return

    run_real(args.dataset, args.model, args.config, device)


if __name__ == "__main__":
    main()
