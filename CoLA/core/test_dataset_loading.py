import sys
import os
import torch
from torch.utils.data import DataLoader

# 确保能找到 core 模块
sys.path.append(os.getcwd())
from core.dataset_xrfv2 import XRFV2Dataset

# [新增] 导入 Config 以便修改状态
from core.config_xrfv2 import cfg


def check_dataset_airpods():
    print("======== 开始测试 XRFV2Dataset (IMU + AirPods) ========")

    # 1. 路径 (确保指向 XRFV2_original)
    data_path = '/home/lipei/XRFV2'
    cfg.TEST_DATA_ROOT = '/home/lipei/WWADL/imu'  # 确保测试集根目录正确
    cfg.DATA_PATH = '/home/lipei/XRFV2'
    cfg.TRAIN_H5_PATH = os.path.join(cfg.DATA_PATH, 'train_data.h5')
    cfg.TEST_DATA_ROOT = '/home/lipei/WWADL/imu'
    cfg.GT_PATH = os.path.join(cfg.DATA_PATH, 'imu_annotations.json')

    # 2. [关键] 强制开启 AirPods
    cfg.USE_AIRPODS = True
    print(f"🔧 Configuration: USE_AIRPODS = {cfg.USE_AIRPODS}")

    num_classes = 30
    class_dict = {f"Action_{i}": i for i in range(num_classes)}
    specific_map = {
        "Walking": 24, "Sitting Down": 7, "Reading": 22,
        "Pouring Water": 1, "Taking Medicine": 5, "Lying Down": 27,
        "Using Phone": 21, "Getting Out of Bed": 25
    }
    class_dict.update(specific_map)

    # --- Test 1: Train Mode (H5 大文件联合读取) ---
    print("\n[Step 1] 测试 Train Mode (H5 内读取 IMU+AirPods)...")
    try:
        train_ds = XRFV2Dataset('train', 'imu', 2048, class_dict)
        t_data, t_label, _, t_vid, _ = train_ds[0]

        # 检查维度
        if t_data.shape == (2048, 36):
            print(f"  ✅ Train Data Shape: {t_data.shape} (IMU:30 + AirPods:6)")
        else:
            print(f"  ❌ Train Data Shape Error: {t_data.shape} (Expect [2048, 36])")
    except Exception as e:
        print(f"  ❌ Train Mode Failed: {e}")

    # --- Test 2: Test Mode (单文件动态拼接) ---
    print("\n[Step 2] 测试 Test Mode (跨文件夹拼接 IMU+AirPods)...")
    try:
        test_ds = XRFV2Dataset('test', 'imu', 2048, class_dict)
        print(f"  => Found {len(test_ds)} test files.")

        target_vid = "0_1_3"
        found = False

        for i in range(len(test_ds)):
            data, label, _, vid, length = test_ds[i]

            if target_vid in vid:
                print(f"  ✅ Found Video: {vid}")
                print(f"     Original Length: {length} (Expect ~3400)")

                # 1. 检查总维度
                if data.shape == (2048, 36):
                    print(f"     ✅ Data Shape: {data.shape} (拼接成功!)")
                else:
                    print(f"     ❌ Data Shape Error: {data.shape}")

                # 2. [深度检查] AirPods 数据是否为空 (防止路径错误导致全 0)
                imu_part = data[:, :30]
                airpods_part = data[:, 30:]

                print(f"     IMU 部分绝对值之和: {imu_part.abs().sum().item():.2f}")
                print(f"     AirPods 部分绝对值之和: {airpods_part.abs().sum().item():.2f}")

                if airpods_part.abs().sum().item() < 1e-5:
                    print(
                        "     ❌ 警告: AirPods 数据全为 0！请检查 /home/lipei/WWADL/AirPodsPro(或 airpods) 目录下是否存在该文件。")
                else:
                    print("     ✅ AirPods 数据有效 (非零)。")

                found = True
                break

        if not found:
            print(f"  ⚠️ Video {target_vid} not found in test.csv list.")

    except Exception as e:
        print(f"  ❌ Test Mode Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    check_dataset_airpods()