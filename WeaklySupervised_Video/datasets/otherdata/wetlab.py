from OtherData.WETLAB.dataset_wetlab_ws import WeaklyWetlabDataset


def build_dataset(config: dict, mode: str):
    dataset_cfg = config.get("dataset", {})
    return WeaklyWetlabDataset(
        dataset_dir=config.get("paths", {}).get("data_dir"),
        loso_json=dataset_cfg.get("loso_json", "loso_sbj_0.json"),
        mode=mode,
        fps=dataset_cfg.get("fps", 50),
        num_sensors=dataset_cfg.get("num_sensors", 6),
        num_classes=dataset_cfg.get("num_classes", 14),
        clip_sec=dataset_cfg.get("clip_sec", 1000.0),
        clip_overlap=dataset_cfg.get("clip_overlap", 0.5),
    )
