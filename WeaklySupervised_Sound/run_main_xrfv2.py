import os
from random import random

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import copy
import numpy as np

from train_epoch import train_wsddn_imu, train_pcl_imu
from test_epoch import test_wsddn_imu, test_pcl_imu

TRAIN_FUNCS = {
    "wsddn": train_wsddn_imu,
    "pcl_imu": train_pcl_imu,
    "oicr_imu": train_pcl_imu,
}

TEST_FUNCS = {
    "wsddn": test_wsddn_imu,
    "pcl_imu": test_pcl_imu,
    "oicr_imu": test_pcl_imu,
}

def train(config, exp_name="default"):
    model_type = config["model"]["type"]
    if model_type not in TRAIN_FUNCS:
        raise ValueError(f"Unknown model.type: {model_type}")

    return TRAIN_FUNCS[model_type](config, exp_name=exp_name)

def test(config, checkpoint_path,test_mode="test_window"):
    model_type = config["model"]["type"]

    if model_type not in TEST_FUNCS:
        raise ValueError(f"Unknown model.type: {model_type}")

    return TEST_FUNCS[model_type](config, checkpoint_path, test_mode)



def main():
    base_config = {
        "path": {
            "train_dataset_path": "/home/lipei/XRFV2/",
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/xrfv2",
            "result_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2"
        },
        "model": {
            "type": "wsddn",

            "pretrained_name": "CNN1D",

            "wsddn": {
                # SPP
                "spp_levels": [1, 2, 4],
                "spp_pool": "max",  # "max" or "avg"
            },


            # ==== PCL / OICR  ====
            "pcl": {
                # "roi_head": "tsse",  # "mlp" | "tsse" | "mamba" | "tsse_mamba" |"transformer" | "lstm"

                "refine_times": 3,

                "fg_thresh": 0.5,
                "bg_thresh": 0.1,

                "use_pcl": True,
                "graph_iou_thresh": 0.5,
                "max_pc_num": 3,

                "adapter": {
                    "enable": False,
                    "bottleneck": 128,
                    "kernel_size": 3,
                    "dropout": 0.1,
                    "scale": 0.1,
                    "use_dwconv": True
                },
            },
        },
        "training": {
            "train_backbone": False,
            "backbone_lr": 1e-5,
            "num_proposals": 60,
            "batch_size": 1,
            "num_epochs": 60,
            "lr": 1e-4,
            "lr_step_size": 10,
            "lr_gamma": 0.5,
            "loc_loss_weight": 0.5,
            "num_workers": 4,
            "spatial_reg_iou": 0.8,
            "num_classes": 30,
            "use_airpods": True,
        },
        "testing": {
            "num_proposals_full": 300,
            "num_proposals_window": 80,
            "conf_thresh": 0.05,
            "nms_sigma": 0.3,
            "top_k": 10,
            "device_keep_list": None,
        }
    }

    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    required_files = [
        base_config["path"]["mapping_path"],
        os.path.join(base_config["path"]["train_dataset_path"], "global_stats.json"),
        os.path.join(base_config["path"]["test_dataset_path"], "info.json"),
        os.path.join(base_config["path"]["test_dataset_path"], "imu_annotations.json")
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file missing: {file_path}")

    os.makedirs(base_config["path"]["checkpoint_path"], exist_ok=True)
    os.makedirs(base_config["path"]["result_path"], exist_ok=True)

    experiments = [
        # wsddn_model
        # {"exp_name": "xrfv2_cnn_wsddn", "spatial_reg_weight":1, "model_type": "wsddn"},

        # {"exp_name": "xrfv2_PretrainedCNN_wsddn", "spatial_reg_weight": 1.0, "model_type": "wsddn"},
        # {"exp_name": "xrfv2_PretrainCNN_pcl", "spatial_reg_weight": 0.0, "model_type": "pcl_imu"},
        # {"exp_name": "xrfv2_PretrainCNN_oicr", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},

        # {"exp_name": "xrfv2_noPretrainedCNN_wsddn", "spatial_reg_weight": 1.0, "model_type": "wsddn"},
        # {"exp_name": "xrfv2_noPretrainCNN_pcl", "spatial_reg_weight": 0.0, "model_type": "pcl_imu"},
        # {"exp_name": "xrfv2_noPretrainCNN_oicr", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},

        # {"exp_name": "xrfv2_LoadAndTrainCNN_wsddn", "spatial_reg_weight": 1.0, "model_type": "wsddn"},
        {"exp_name": "xrfv2_LoadAndTrainCNN_pcl", "spatial_reg_weight": 0.0, "model_type": "pcl_imu"},
        # {"exp_name": "xrfv2_LoadAndTrainCNN_oicr", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},


    ]

    ckpt_paths = {}

    for exp in experiments:
        exp_name = exp["exp_name"]
        lam = exp["spatial_reg_weight"]
        model_type = exp["model_type"]

        config = copy.deepcopy(base_config)
        config["training"]["spatial_reg_weight"] = lam
        config["model"]["type"] = exp["model_type"]

        result_root = base_config["path"]["result_path"]
        exp_result_path = os.path.join(result_root, exp_name)
        config["path"]["result_path"] = exp_result_path
        os.makedirs(exp_result_path, exist_ok=True)

        print(f"\nStarting experiment: {exp_name} (model_type={model_type}, spatial_reg_weight={lam})")

        ckpt_path = train(config, exp_name=exp_name)
        # ckpt_path = "/home/yangzhenkui/code/WSDDN/checkpoints/wsddn_transformer_spatial_reg.pth"
        ckpt_paths[exp_name] = ckpt_path



        print(f"\nTraining finished, starting evaluation: {exp_name}")

        test(config, ckpt_path,"test_full")
        test(config, ckpt_path,"test_window")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nProgram error: {str(e)}")
        import traceback
        traceback.print_exc()
