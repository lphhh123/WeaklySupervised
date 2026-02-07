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
    "wsddn_avg": train_wsddn_imu,
    "wsddn": train_wsddn_imu,
    "wsddn_transformer": train_wsddn_imu,
    "pcl_imu": train_pcl_imu,
    "oicr_imu": train_pcl_imu,
}

TEST_FUNCS = {
    "wsddn_avg": test_wsddn_imu,
    "wsddn": test_wsddn_imu,
    "wsddn_transformer": test_wsddn_imu,
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

    return TEST_FUNCS[model_type](config, checkpoint_path,test_mode)




def main():
    base_config = {
        "path": {
            "train_dataset_path": "/home/lipei/XRFV2/",
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "/home/lipei/project/WSDDN/checkpoints/xrfv2/2022",
            "result_path": "/home/lipei/project/WSDDN/test_results/xrfv2/2022"
        },
        "model": {


            "type": "wsddn",


            "pretrained_name": "CNN1D",


            "wsddn": {

                "d_model": 512,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 1024,
                "dropout": 0.1,
                "use_positional_encoding": True,


                "spp_levels": [1, 2, 4],
                "spp_pool": "max",  # "max" or "avg"

            },


            # ==== PCL / OICR  ====
            "pcl": {
                # "roi_head": "tsse",  # "mlp" | "tsse" | "mamba" | "tsse_mamba" |"transformer" | "lstm"


                "refine_times": 3,


                "fg_thresh": 0.5,
                "bg_thresh": 0.1,


                "use_pcl": False,
                "graph_iou_thresh": 0.5,
                "max_pc_num": 3,

            },
        },
        "training": {
            "train_backbone": False,
            "backbone_lr": 1e-5,
            "num_proposals": 60,
            "batch_size": 16,
            "num_epochs": 80,
            "lr": 1e-4,
            "lr_step_size": 10,
            "lr_gamma": 0.9,
            "loc_loss_weight": 0.5,
            "num_workers": 4,
            "spatial_reg_iou": 0.8,
            "num_classes": 30,
            "use_airpods": True,
        },
        "testing": {
            "num_proposals_full":2000,
            "num_proposals_window": 80,
            "conf_thresh": 0.02,
            "nms_sigma": 0.3,
            "top_k": 10,
            "device_keep_list": None,

            "confusion_tiou": 0.5
        }
    }


    import random as pyrandom

    pyrandom.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)


    required_files = [
        base_config["path"]["mapping_path"],
        os.path.join(base_config["path"]["train_dataset_path"], "global_stats.json"),
        os.path.join(base_config["path"]["test_dataset_path"], "info.json"),
        os.path.join(base_config["path"]["test_dataset_path"], "imu_annotations.json")
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Message：{file_path}")

    os.makedirs(base_config["path"]["checkpoint_path"], exist_ok=True)
    os.makedirs(base_config["path"]["result_path"], exist_ok=True)


    experiments = [
        # wsddn_model
        {"exp_name": "xrfv2_cnn_wsddn_01", "spatial_reg_weight":1, "model_type": "wsddn"},


        # {"exp_name": "xrfv2_cnn_pcl_0112", "spatial_reg_weight": 0.0, "model_type": "pcl_imu"},



        # {"exp_name": "xrfv2_full_window_0109", "spatial_reg_weight": 1.0, "model_type": "wsddn"},
        # {"exp_name": "xrfv2_oicr_full_window_0109", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},


        # {"exp_name": "person2_xrfv2_cnn_wsddn_0120", "spatial_reg_weight":1, "model_type": "wsddn"},
        # {"exp_name": "person2_xrfv2_cnn_oicr_0120", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},

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

        print("\n" + "=" * 60)
        print(f"Message：{exp_name}（model_type={model_type}, spatial_reg_weight={lam}）")
        print("=" * 60)


        ckpt_path = train(config, exp_name=exp_name)
        # ckpt_path = "/home/yangzhenkui/code/WSDDN/checkpoints/wsddn_transformer_spatial_reg.pth"
        ckpt_paths[exp_name] = ckpt_path



        print("\n" + "=" * 60)
        print(f"Message，Message：{exp_name}")
        print("=" * 60)


        test(config, ckpt_path,"test_full")
        test(config, ckpt_path,"test_window")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nMessage：{str(e)}")
        import traceback
        traceback.print_exc()
