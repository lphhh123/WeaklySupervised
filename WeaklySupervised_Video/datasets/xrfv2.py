from types import SimpleNamespace

from dataset.dataset_xrfv2 import (
    WeaklySupervisedXRFV2DatasetTrain,
    WeaklySupervisedXRFV2DatasetTest,
)
from models.cola.dataset_xrfv2 import XRFV2Dataset as ColaXRFV2Dataset


def build_rskp_xrfv2_train(config: dict):
    paths = config.get("paths", {})
    return WeaklySupervisedXRFV2DatasetTrain(
        dataset_dir=paths.get("data_dir"),
        mapping_path=paths.get("mapping_path"),
        use_airpods=config.get("dataset", {}).get("use_airpods", True),
    )


def build_rskp_xrfv2_test(config: dict):
    dataset_cfg = {
        "path": {
            "test_dataset_path": config.get("paths", {}).get("data_dir"),
            "mapping_path": config.get("paths", {}).get("mapping_path"),
        }
    }
    return WeaklySupervisedXRFV2DatasetTest(
        config=dataset_cfg,
        modality="imu",
        device_keep_list=None,
        use_airpods=config.get("dataset", {}).get("use_airpods", True),
    )


def build_cola_xrfv2_dataset(config: dict, mode: str):
    cfg = SimpleNamespace(
        data_path=config.get("paths", {}).get("data_dir"),
        gt_path=config.get("dataset", {}).get("gt_path"),
        test_data_root=config.get("dataset", {}).get("test_data_root"),
        use_airpods=config.get("dataset", {}).get("use_airpods", False),
    )
    return ColaXRFV2Dataset(
        mode=mode,
        modal=config.get("dataset", {}).get("modal", "imu"),
        num_segments=config.get("dataset", {}).get("num_segments", 2048),
        class_dict=config.get("dataset", {}).get("class_dict", {}),
        cfg=cfg,
    )
