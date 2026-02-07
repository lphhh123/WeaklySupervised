import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm


sys.path.append(os.getcwd())

from core.model import CoLA

from run_cola_hangtime_ddp import map_config_to_cola_cfg, run_inference_dual_mode


RESULTS_ROOT = "./output_hangtime_cola_ddp"
SEEDS = [2022, 2024, 2026]
FOLDS = range(24)

BASE_CONFIG = {
    "dataset_dir": "/home/lipei/TAL_data/hangtime/",
    "pretrained_dir": "/home/lipei/project/WSDDN/OtherData/HANGTIME/pre_train/CNN1D",
    "result_root": RESULTS_ROOT,
    "fps": 50,
    "in_channels": 3,
    "num_classes": 5,
    "clip_sec": 30.0,
    "clip_overlap": 0.5,
    "pretrained_model_name": "CNN1D",
    "training": {
        "train_backbone": False
    },
    "cola": {
        "lambda": 0.01,
        "r_easy": 100,
        "r_hard": 20,
        "m": 3,
        "M": 10,
        "class_thresh": 0,
        "nms_thresh": 0.4
    }
}


# ===============================================================

def run_all_re_inference():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for seed in SEEDS:
        print(f"\n" + "=" * 30)
        print(f"🚀 Message SEED: {seed}")
        print("=" * 30)


        config = copy.deepcopy(BASE_CONFIG)
        config["result_root"] = os.path.join(BASE_CONFIG["result_root"], f"seed_{seed}")

        for fold in FOLDS:
            fold_dir = os.path.join(config["result_root"], f"fold{fold}")
            ckpt_path = os.path.join(fold_dir, "model_final.pth")

            if not os.path.exists(ckpt_path):
                print(f"   ⚠️ Message Fold {fold}: Message")
                continue

            print(f"   -> Processing Fold {fold}...")



            try:
                run_inference_dual_mode(config, fold, ckpt_path, device)
            except Exception as e:
                print(f"   ❌ Fold {fold} Message: {e}")

    print("\n" + "=" * 50)
    print("✅ Message！")
    print(f"Message: {RESULTS_ROOT}")
    print("=" * 50)


if __name__ == "__main__":

    import copy

    run_all_re_inference()