import argparse
import os

import torch

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
import numpy as np

from train_epoch import train_wsddn_imu, train_pcl_imu
from test_epoch import test_wsddn_imu, test_pcl_imu
from scripts.config_utils import load_json_config

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


def test(config, checkpoint_path, test_mode="test_window"):
    model_type = config["model"]["type"]

    if model_type not in TEST_FUNCS:
        raise ValueError(f"Unknown model.type: {model_type}")

    return TEST_FUNCS[model_type](config, checkpoint_path, test_mode)


def main():
    parser = argparse.ArgumentParser(description="XRFV2 training/testing entry")
    parser.add_argument("--config", default="configs/xrfv2_wsddn.json", help="Path to JSON config")
    args = parser.parse_args()

    base_config, _ = load_json_config(args.config)

    import random as pyrandom

    pyrandom.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)

    required_files = [
        base_config["path"]["mapping_path"],
        os.path.join(base_config["path"]["train_dataset_path"], "global_stats.json"),
        os.path.join(base_config["path"]["test_dataset_path"], "info.json"),
        os.path.join(base_config["path"]["test_dataset_path"], "imu_annotations.json"),
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file missing: {file_path}")

    os.makedirs(base_config["path"]["checkpoint_path"], exist_ok=True)
    os.makedirs(base_config["path"]["result_path"], exist_ok=True)

    exp_name = base_config.get("exp_name", "xrfv2_run")
    print(f"\nStarting experiment: {exp_name} (model_type={base_config['model']['type']})")
    ckpt_path = train(base_config, exp_name=exp_name)

    print(f"\nTraining complete, starting test: {exp_name}")
    test(base_config, ckpt_path, "test_full")
    test(base_config, ckpt_path, "test_window")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nProgram error: {str(e)}")
        import traceback

        traceback.print_exc()
