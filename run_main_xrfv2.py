import os
from random import random

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import copy
import numpy as np

from train_epoch import train_wscnet_imu, train_wsddn_imu, train_pcl_imu, train_pclHead, train_oicrPALoss_imu
from test_epoch import test_wscnet_imu, test_wsddn_imu, test_pcl_imu, test_pclHead, test_oicrPALoss_imu

TRAIN_FUNCS = {
    "wsddn_avg": train_wsddn_imu,
    "wsddn": train_wsddn_imu,
    "wsddn_transformer": train_wsddn_imu,
    "wscnet_imu": train_wscnet_imu,
    "pcl_imu": train_pcl_imu,
    "oicr_imu": train_pcl_imu,
    "wsddn_mamba_basic": train_wsddn_imu,
    "wsddn_mamba_bi": train_wsddn_imu,
    "pclHead":train_pclHead,
    "oicr_pa_imu" : train_oicrPALoss_imu,
}

TEST_FUNCS = {
    "wsddn_avg": test_wsddn_imu,
    "wsddn": test_wsddn_imu,
    "wsddn_transformer": test_wsddn_imu,
    "wscnet_imu": test_wscnet_imu,
    "pcl_imu": test_pcl_imu,
    "oicr_imu": test_pcl_imu,
    "wsddn_mamba_basic": test_wsddn_imu,
    "wsddn_mamba_bi": test_wsddn_imu,
    "pclHead": test_pclHead,
    "oicr_pa_imu" : test_oicrPALoss_imu,
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



# ========================== 主函数：跑多个实验 ==========================
def main():
    base_config = {
        "path": {
            "train_dataset_path": "/home/lipei/XRFV2/",
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "/home/lipei/project/WSDDN/checkpoints/xrfv2",
            "result_path": "/home/lipei/project/WSDDN/test_results/xrfv2"
        },
        "model": {
            # ======== 通用部分 ========
            # 模型名称：这里给一个默认值，真正用哪个看 experiments 里的 model_type
            "type": "wsddn",

            # 特征提取模块（预训练模型）："CNN1D"、"TSSE_Mamba"、"Mamba"、"TSSE"、"VGG1D_BUAA"
            "pretrained_name": "VGG1D_BUAA",

            # ======== WSDDN 系列专用配置 ========
            "wsddn": {
                # Transformer 版 WSDDN 的配置
                "d_model": 512,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 1024,
                "dropout": 0.1,
                "use_positional_encoding": True,

                # SPP 版 WSDDN 的配置
                "spp_levels": [1, 2, 4],  # 多尺度池化层数
                "spp_pool": "max",  # "max" or "avg"

                # mamba
                "mamba_hidden_dim": 512,
                "mamba_depth": 3,
                "mamba_dropout": 0.1,
                "mamba_share_weights": True,
                "mamba_merge_mode": "sum",
            },


            # ==== PCL / OICR  ====
            "pcl": {
                # "roi_head": "tsse",  # "mlp" | "tsse" | "mamba" | "tsse_mamba" |"transformer" | "lstm"

                # 共用：PCL / OICR 的 refine 次数
                "refine_times": 3,

                # IoU 阈值
                "fg_thresh": 0.5,  # 前景
                "bg_thresh": 0.1,  # 忽略阈值 (max_overlap < bg_thresh -> ignore)

                # PCL 专用
                "use_pcl": False,  # False = 只用 OICR; True = 启用 PCL cluster 逻辑
                "graph_iou_thresh": 0.5, # 如果两个 proposal 的 IoU > 某个阈值，就连一条边
                "max_pc_num": 3,  # 每类最多 cluster center 个数

                # adapter（训练）
                "adapter": {
                    "enable": False,
                    "bottleneck": 128,
                    "kernel_size": 3,
                    "dropout": 0.1,
                    "scale": 0.1,
                    "use_dwconv": True
                },

                # ===== OICR + PA-Loss 配置 =====
                "stage0_boost": 3.0,  # 原论文 trick：第0个refine分支loss乘一个系数；不需要就设为1.0
                "pa_mode": "sigmoid",  # "sigmoid" or "exp"
                "enhance_weight": True,  # 是否做 w = w * exp(w)（强化高分pseudo GT）
            },
        },
        "training": {
            "train_backbone": False,  # false=冻结backbone，只训练head；true=backbone也训练
            "backbone_lr": 1e-5,  # 可选：backbone单独lr（一般比head小）
            "num_proposals": 60,
            "batch_size": 16,
            "num_epochs": 60,
            "lr": 1e-4,
            "lr_step_size": 10,
            "lr_gamma": 0.5,
            "loc_loss_weight": 0.5,
            "num_workers": 4,
            "spatial_reg_iou": 0.8,  # 原wsddn是0.6
            "num_classes": 30,
            "use_airpods": True,
        },
        "testing": {
            "num_proposals_full":300,
            "num_proposals_window": 80,
            "conf_thresh": 0.05,  # 高于这个置信度的作为候选片段
            "nms_sigma": 0.3,
            "top_k": 10,
            "device_keep_list": None,

            "confusion_tiou": 0.5  #评价指标混淆矩阵依托的tIoU
        }
    }

    # 固定随机种子
    import random as pyrandom

    pyrandom.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    # 检查必要文件
    required_files = [
        base_config["path"]["mapping_path"],
        os.path.join(base_config["path"]["train_dataset_path"], "global_stats.json"),
        os.path.join(base_config["path"]["test_dataset_path"], "info.json"),
        os.path.join(base_config["path"]["test_dataset_path"], "imu_annotations.json")
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"必要文件缺失：{file_path}")

    os.makedirs(base_config["path"]["checkpoint_path"], exist_ok=True)
    os.makedirs(base_config["path"]["result_path"], exist_ok=True)

    # 定义实验
    experiments = [
        # wsddn_model
        # {"exp_name": "xrfv2_cnn_wsddn_0112", "spatial_reg_weight":1, "model_type": "wsddn"},
        # # 增加transformer
        # {"exp_name": "wsddn_transformer_spatial_reg", "spatial_reg_weight":1, "model_type": "wsddn_transformer"},
        # {"exp_name": "wsddn_transformer_no_reg", "spatial_reg_weight":0.0, "model_type": "wsddn_transformer"},
        # # ssp换成avg
        # {"exp_name": "wsddn_avg_spatial_reg", "spatial_reg_weight": 1, "model_type": "wsddn_avg"},
        # {"exp_name": "wsddn_avg_no_reg", "spatial_reg_weight": 0.0, "model_type": "wsddn_avg"},

        # {"exp_name": "wscnet",  "spatial_reg_weight": 0.0, "model_type": "wscnet_imu"},

        # 注意改"use_pcl": True
        # {"exp_name": "xrfv2_cnn_pcl_0112", "spatial_reg_weight": 0.0, "model_type": "pcl_imu"},
        # 注意改"use_pcl": False
        # {"exp_name": "xrfv2_cnn_oicr_0112", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},

        # mamba
        # {"exp_name": "wsddn_mamba_basic", "spatial_reg_weight": 1, "model_type": "wsddn_mamba_basic"},
        # {"exp_name": "wsddn_mamba_bi", "spatial_reg_weight": 1, "model_type": "wsddn_mamba_bi"},


        # 核心模块：TSSE
        # {"exp_name": "xrfv2_pcl_TSSE", "spatial_reg_weight": 0.0, "model_type": "pclHead"},
        # {"exp_name": "xrfv2_oicr_TSSE", "spatial_reg_weight": 0.0, "model_type": "pclHead"},
        # 核心模块：mamba
        # {"exp_name": "xrfv2_pcl_mamba", "spatial_reg_weight": 0.0, "model_type": "pclHead"},
        # 核心模块：TSSE+mamba
        # {"exp_name": "xrfv2_pcl_TSSE_mamba", "spatial_reg_weight": 0.0, "model_type": "pclHead"},
        # 核心模块：lstm
        # {"exp_name": "xrfv2_pcl_lstm", "spatial_reg_weight": 0.0, "model_type": "pclHead"},
        # 核心模块：transformer
        # {"exp_name": "xrfv2_pcl_transformer", "spatial_reg_weight": 0.0, "model_type": "pclHead"},

        # 原模型（正确预训练模型路径+"train_backbone": True)
        # {"exp_name": "xrfv2_PretrainedCNN_wsddn", "spatial_reg_weight": 1.0, "model_type": "wsddn"},
        # {"exp_name": "xrfv2_PretrainCNN_pcl", "spatial_reg_weight": 0.0, "model_type": "pcl_imu"},
        # {"exp_name": "xrfv2_PretrainCNN_oicr", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},

        # 随机初始化预训练模块并参与后续训练（修改预训练模型路径+"train_backbone": True)
        # {"exp_name": "xrfv2_noPretrainedCNN_wsddn", "spatial_reg_weight": 1.0, "model_type": "wsddn"},
        # {"exp_name": "xrfv2_noPretrainCNN_pcl", "spatial_reg_weight": 0.0, "model_type": "pcl_imu"},
        # {"exp_name": "xrfv2_noPretrainCNN_oicr", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},

        # 加载预训练模块并参与后续训练（正确预训练模型路径+"train_backbone": True)
        # {"exp_name": "xrfv2_LoadAndTrainCNN_wsddn", "spatial_reg_weight": 1.0, "model_type": "wsddn"},
        # {"exp_name": "xrfv2_LoadAndTrainCNN_pcl", "spatial_reg_weight": 0.0, "model_type": "pcl_imu"},
        # {"exp_name": "xrfv2_LoadAndTrainCNN_oicr", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},

        # 修改了generate_box
        # {"exp_name": "xrfv2_full_window_0109", "spatial_reg_weight": 1.0, "model_type": "wsddn"},
        # {"exp_name": "xrfv2_oicr_full_window_0109", "spatial_reg_weight": 0.0, "model_type": "oicr_imu"},

        # oicr_PALoss
        {"exp_name": "xrfv2_oicr_boost3_0113", "spatial_reg_weight": 0.0, "model_type": "oicr_pa_imu"},


    ]

    ckpt_paths = {}

    for exp in experiments:
        exp_name = exp["exp_name"]
        lam = exp["spatial_reg_weight"]
        model_type = exp["model_type"]

        # 拷一份 config，并写入本实验的参数
        config = copy.deepcopy(base_config)
        config["training"]["spatial_reg_weight"] = lam
        config["model"]["type"] = exp["model_type"]

        # 为每个实验设置单独结果目录
        result_root = base_config["path"]["result_path"]
        exp_result_path = os.path.join(result_root, exp_name)
        config["path"]["result_path"] = exp_result_path
        os.makedirs(exp_result_path, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"开始实验：{exp_name}（model_type={model_type}, spatial_reg_weight={lam}）")
        print("=" * 60)

        # 这里用统一的 train 分发：会根据 model.type 调 wsddn / wscnet
        ckpt_path = train(config, exp_name=exp_name)
        # ckpt_path = "/home/yangzhenkui/code/WSDDN/checkpoints/wsddn_transformer_spatial_reg.pth"
        ckpt_paths[exp_name] = ckpt_path



        print("\n" + "=" * 60)
        print(f"训练完成，开始测试：{exp_name}")
        print("=" * 60)

        # test 同样分发
        test(config, ckpt_path,"test_full")
        test(config, ckpt_path,"test_window")


# ========================== 程序入口 ==========================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序运行出错：{str(e)}")
        import traceback
        traceback.print_exc()
