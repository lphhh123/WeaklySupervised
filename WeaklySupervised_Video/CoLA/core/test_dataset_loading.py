import os
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from models.cola.dataset_xrfv2 import XRFV2Dataset


def check_dataset_airpods():
    print("Starting XRFV2Dataset test (IMU + AirPods)")

                                 
    data_path = os.environ.get("XRFV2_DATA_PATH", os.path.join(os.getcwd(), "data", "XRFV2"))
    test_root = os.environ.get("XRFV2_TEST_ROOT", os.path.join(os.getcwd(), "data", "WWADL", "imu"))
    cfg = SimpleNamespace(
        data_path=data_path,
        test_data_root=test_root,
        gt_path=os.path.join(data_path, "imu_annotations.json"),
        use_airpods=True,
    )

                          
    print(f"Configuration: USE_AIRPODS = {cfg.use_airpods}")

    num_classes = 30
    class_dict = {f"Action_{i}": i for i in range(num_classes)}
    specific_map = {
        "Walking": 24, "Sitting Down": 7, "Reading": 22,
        "Pouring Water": 1, "Taking Medicine": 5, "Lying Down": 27,
        "Using Phone": 21, "Getting Out of Bed": 25
    }
    class_dict.update(specific_map)

                                             
    print("\n[Step 1] Test Train Mode (read IMU+AirPods from H5)...")
    try:
        train_ds = XRFV2Dataset('train', 'imu', 2048, class_dict, cfg)
        t_data, t_label, _, t_vid, _ = train_ds[0]

              
        if t_data.shape == (2048, 36):
            print(f"  Train Data Shape: {t_data.shape} (IMU:30 + AirPods:6)")
        else:
            print(f"  Train Data Shape Error: {t_data.shape} (Expect [2048, 36])")
    except Exception as e:
        print(f"  Train Mode Failed: {e}")

                                         
    print("\n[Step 2] Test Test Mode (merge IMU+AirPods across folders)...")
    try:
        test_ds = XRFV2Dataset('test', 'imu', 2048, class_dict, cfg)
        print(f"  => Found {len(test_ds)} test files.")

        target_vid = "0_1_3"
        found = False

        for i in range(len(test_ds)):
            data, label, _, vid, length = test_ds[i]

            if target_vid in vid:
                print(f"  Found Video: {vid}")
                print(f"     Original Length: {length} (Expect ~3400)")

                          
                if data.shape == (2048, 36):
                    print(f"     Data Shape: {data.shape} (merge successful)")
                else:
                    print(f"     Data Shape Error: {data.shape}")

                                                        
                imu_part = data[:, :30]
                airpods_part = data[:, 30:]

                print(f"     IMU absolute sum: {imu_part.abs().sum().item():.2f}")
                print(f"     AirPods absolute sum: {airpods_part.abs().sum().item():.2f}")

                if airpods_part.abs().sum().item() < 1e-5:
                    print(
                        "     Warning: AirPods data is all zeros. Check that the file exists under "
                        "data/WWADL/AirPodsPro (or airpods)."
                    )
                else:
                    print("     AirPods data is valid (non-zero).")

                found = True
                break

        if not found:
            print(f"  Warning: Video {target_vid} not found in test.csv list.")

    except Exception as e:
        print(f"  Test Mode Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    check_dataset_airpods()
