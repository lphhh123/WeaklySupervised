import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm

# 引入项目路径
sys.path.append(os.getcwd())

from core.model import CoLA
# 直接从你之前的 DDP 脚本中导入已经写好的核心推理函数
from run_cola_hangtime_ddp import map_config_to_cola_cfg, run_inference_dual_mode

# ================= 配置区域：必须与训练时完全一致 =================
RESULTS_ROOT = "./output_hangtime_cola_ddp"
SEEDS = [2022, 2024, 2026]
FOLDS = range(24)

BASE_CONFIG = {
    "dataset_dir": "/home/lipei/TAL_data/hangtime/",
    "pretrained_dir": "/home/lipei/project/WSDDN/OtherData/HANGTIME/pre_train/CNN1D",
    "result_root": RESULTS_ROOT,  # 根目录
    "fps": 50,
    "in_channels": 3,
    "num_classes": 5,
    "clip_sec": 30.0,
    "clip_overlap": 0.5,
    "pretrained_model_name": "CNN1D",
    "training": {
        "train_backbone": False  # 推理时设为 False
    },
    "cola": {
        "lambda": 0.01,
        "r_easy": 100,
        "r_hard": 20,
        "m": 3,
        "M": 10,
        "class_thresh": 0,
        "nms_thresh": 0.4  # 如果你之后想调 NMS 阈值，改这里就行
    }
}


# ===============================================================

def run_all_re_inference():
    # 使用单张显卡即可，速度非常快
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for seed in SEEDS:
        print(f"\n" + "=" * 30)
        print(f"🚀 开始重推理 SEED: {seed}")
        print("=" * 30)

        # 这一步是为了让 map_config_to_cola_cfg 能正确找到权重所在目录
        config = copy.deepcopy(BASE_CONFIG)
        config["result_root"] = os.path.join(BASE_CONFIG["result_root"], f"seed_{seed}")

        for fold in FOLDS:
            fold_dir = os.path.join(config["result_root"], f"fold{fold}")
            ckpt_path = os.path.join(fold_dir, "model_final.pth")

            if not os.path.exists(ckpt_path):
                print(f"   ⚠️ 跳过 Fold {fold}: 权重文件不存在")
                continue

            print(f"   -> Processing Fold {fold}...")

            # 调用你 DDP 脚本里的推理逻辑
            # 该逻辑包含：加载模型、滑窗/插值、Sigmoid 规范化分数、保存 JSON
            try:
                run_inference_dual_mode(config, fold, ckpt_path, device)
            except Exception as e:
                print(f"   ❌ Fold {fold} 推理失败: {e}")

    print("\n" + "=" * 50)
    print("✅ 所有种子的双模式推理已完成！")
    print(f"请查看目录: {RESULTS_ROOT}")
    print("=" * 50)


if __name__ == "__main__":
    # 为了防止 copy.deepcopy 报错，引入 copy
    import copy

    run_all_re_inference()