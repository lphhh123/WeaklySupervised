import argparse
import os
import random

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import copy
import numpy as np

from train_epoch import train_wsddn_imu, train_pcl_imu, train_rskp_imu
from test_epoch import test_wsddn_imu, test_pcl_imu, test_rskp_imu

TRAIN_FUNCS = {
    "wsddn": train_wsddn_imu,
    "pcl_imu": train_pcl_imu,
    "oicr_imu": train_pcl_imu,
    "rskp": train_rskp_imu,
}

TEST_FUNCS = {
    "wsddn": test_wsddn_imu,
    "pcl_imu": test_pcl_imu,
    "oicr_imu": test_pcl_imu,
    "rskp": test_rskp_imu,
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



# ========================== 主函数：跑多个实验 ==========================
def main():
    parser = argparse.ArgumentParser(description="RSKP、Training")
    parser.add_argument("--seed", type=int, default=2024, help="seed (e.g., 0, 1, 2)")
    parser.add_argument("--person", type=str, default='person0', help="seed (e.g., 0, 1, 2)")
    args = parser.parse_args()
    datasets_config = {
      'person0' : {
          "train_dataset_path": "/home/lipei/XRFV2/",
          "test_dataset_path": "/home/lipei/XRFV2/",
          'exp_name':'xrfv2_CNN_rskp_0130',
          "pretrained_name": "CNN1D"
      },
      'person2' : {
          "train_dataset_path": "/data1/WSTAL/all_2_30_3/",
          "test_dataset_path": "/data1/WSTAL/all_2_30_3/",
          'exp_name':'xrfv2_CNN_rskp_person2_0130',
          "pretrained_name": "CNN1D_person2"
      },
      'person6' : {
          "train_dataset_path": "/data1/WSTAL/all_6_30_3/",
          "test_dataset_path": "/data1/WSTAL/all_6_30_3/",
          'exp_name':'xrfv2_CNN_rskp_person6_0130',
          "pretrained_name": "CNN1D_person6"
      },
      'person5': {
          "train_dataset_path": "/data1/WSTAL/all_5_30_3/",
          "test_dataset_path": "/data1/WSTAL/all_5_30_3/",
          'exp_name':'xrfv2_CNN_rskp_person5_0130',
          "pretrained_name": "CNN1D_person5"
      },
      'person4': {
          "train_dataset_path": "/data/WSTAL/all_4_30_3/",
          "test_dataset_path": "/data/WSTAL/all_4_30_3/",
          'exp_name':'xrfv2_CNN_rskp_person4_0130',
          "pretrained_name": "CNN1D_person4"
      }
    }
    seed = args.seed
    config_set = datasets_config[args.person]
    base_config = {
        "path": {
            "train_dataset_path": config_set["train_dataset_path"],
            "test_dataset_path": config_set["test_dataset_path"],
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": f"/home/lipei/project/WSDDN/checkpoints/xrfv2/{seed}",
            "result_path": f"/home/lipei/project/WSDDN/test_results/xrfv2/{seed}"
        },
        "model": {
            # ======== 通用部分 ========
            # 模型名称：这里给一个默认值，真正用哪个看 experiments 里的 model_type
            "type": "rskp",

            # 特征提取模块（预训练模型）："CNN1D"、"TSSE_Mamba"、"Mamba"、"TSSE"
            "pretrained_name": config_set["pretrained_name"],

            # ======== WSDDN 系列专用配置 ========
            "wsddn": {
                # SPP
                "spp_levels": [1, 2, 4],  # 多尺度池化层数
                "spp_pool": "max",  # "max" or "avg"
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
                "use_pcl": True,  # False = 只用 OICR; True = 启用 PCL cluster 逻辑
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
            },

            # ==== RSKP ====
            "rskp": {
                # 特征维度（可选，默认使用 backbone 输出维度）
                "out_feat_num": None,  # None 表示自动从 backbone 获取

                # Random Walk 参数
                "w": 0.2,  # random walk 权重

                # EM 迭代参数
                "mu_num": 8,  # 每个样本的 representative snippet 数量（Gaussian tokens）
                "em_iter": 3,  # EM 迭代次数

                # 记忆库参数
                "mu_queue_len": 8,  # 每个类别的记忆库队列长度

                # 模型参数
                "scale_factor": 4.0,  # 温度缩放因子
                "dropout": 0.4,  # dropout 率
                "T": 0.2,  # CategoryCrossEntropy 的温度参数

                # 损失权重
                "lambda_a": 0.05,  # attention normalization loss 权重
                "lambda_b": 0.2,  # class-wise attention branch 权重
                "lambda_s": 1.0,  # pseudo label supervision loss 权重

                # Warmup 参数
                "warmup_epoch": 60,  # 从第几个 epoch 开始启用记忆库和 random walk

                # 测试时融合权重
                "test_fusion_weight": 0.6,  # 测试时原始特征分支权重
            },
        },
        "training": {
            "train_backbone": True,  # false=冻结backbone，只训练head；true=backbone也训练
            "backbone_lr": 1e-5,  # 可选：backbone单独lr（一般比head小）
            "num_proposals": 60,
            "batch_size": 32,
            "num_epochs": 80,
            "lr": 1e-4,
            "lr_step_size": 10,
            "lr_gamma": 0.9,
            "loc_loss_weight": 0.5,
            "num_workers": 4,
            "spatial_reg_iou": 0.8,  # 原wsddn是0.6
            "num_classes": 30,
            "use_airpods": True,
        },
        "testing": {
            "num_proposals_full": 300,
            "num_proposals_window": 80,
            "conf_thresh": 0.01,  # 高于这个置信度的作为候选片段
            "nms_sigma": 0.3,
            "top_k": 200,
            "device_keep_list": None,
        }
    }

    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        # {"exp_name": "xrfv2_cnn_wsddn", "spatial_reg_weight":1, "model_type": "wsddn"},

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

        # RSKP 模型
        {"exp_name": config_set["exp_name"], "spatial_reg_weight": 0.0, "model_type": "rskp"},

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

        # 这里用统一的 train 分发：会根据 model.type
        ckpt_path = train(config, exp_name=exp_name)
        # ckpt_path = "/home/yangzhenkui/code/WSDDN/checkpoints/wsddn_transformer_spatial_reg.pth"
        ckpt_paths[exp_name] = ckpt_path



        print("\n" + "=" * 60)
        print(f"训练完成，开始测试：{exp_name}")
        print("=" * 60)

        # test 同样分发
        test(config, ckpt_path,"test_full")  # 整条式
        test(config, ckpt_path,"test_window")  # window式

# ========================== 程序入口 ==========================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序运行出错：{str(e)}")
        import traceback
        traceback.print_exc()
