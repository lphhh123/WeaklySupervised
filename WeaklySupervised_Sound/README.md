# WeaklySupervisedTemporalActionLocalization (IMU)

This repository contains research/experimental code for **Weakly-Supervised Temporal Action Localization (WS-TAL)**, focused on **wearable/IMU 1D sensor sequences** (IMU with optional AirPods sensor concatenation).

Current main implementations:

- **WSDDN-IMU**: WSDDN dual-branch (classification + detection) for weak localization; proposal pooling uses **1D Temporal SPP** (`TemporalSPP1D`).
- **PCL/OICR-IMU**: Adds **OICR refine** on top of WSDDN/MIL, with optional **PCL (Proposal Cluster Learning)**; internally uses **KMeans** to generate pseudo GT / cluster centers.
- **Inference & evaluation**: Sliding-window inference on long sequences, per-class **Soft-NMS**, and ActivityNet-style `ANETdetection` for **mAP@tIoU**.

---

## 1. Environment & installation
```bash
pip install -r requirements.txt
```

---

## 2. Code structure (core files)
The project has two parts: the XRFV2 dataset and other datasets. XRFV2 training/testing code lives in the root folder, while other datasets are under `OtherData/` (their structures are consistent).

Datasets:
- XRFV2: `./dataset/dataset_xrfv2.py`
- Other datasets: `dataset_{dataset_name}_ws.py` in each dataset folder

Core scripts:
- `run_main_xrfv2.py`: **XRFV2** main entry (config is defined in the script and it chains training + testing).
- `train_epoch.py`: training loop
  - `train_wsddn_imu(config)`
  - `train_pcl_imu(config)`
- `test_epoch.py`: testing/inference and evaluation
  - `test_wsddn_imu(config, checkpoint_path)`
  - `test_pcl_imu(config, checkpoint_path)`
- `dataset/dataset_xrfv2.py`: XRFV2/WWADL-style datasets (training with `*.h5`, testing with `test.csv` + per-file h5)
- `models/WSDDN_model.py`: WSDDN + TemporalSPP1D + proposal generation
- `models/PCL_OICR_model.py`: PCL/OICR heads (KMeans + refine)
- `builder_pretrainbackbone.py`: register/load pretrained backbones (`PRETRAINED_ZOO`)
- `tool.py`: Soft-NMS, ANETdetection (mAP), etc.

### 2.1 Other datasets

`OtherData/` provides weakly supervised training/testing scripts for multiple datasets (e.g., `Opportunity/`, `RWHAR/`, `HANGTIME/`, `WEAR/`, `WETLAB/`, `SBHAR/`). The workflow is similar to the root, but data organization and LOSO splits vary.

---

## 3. Data preparation (XRFV2 example)

### 3.1 Training set directory (`train_dataset_path`)

Training uses weak supervision at the 30s clip level. Expected files:

- `train_data.h5`
  - `imu`: shape **[N, 2048, 5, 6]** (2048 samples for 30s; 5 devices; 6 dims/device)
  - `airpods` (optional): shape **[N, 2048, 9]** (uses AirPods acc(3) + gyro(3))
- `train_label.json`
  - Includes `imu` field: `{sample_idx(str): [ [left_offset, right_offset, old_label_id], ... ] }`
  - `left_offset/right_offset` are relative positions
- `global_stats.json`
  - Global mean and variance per modality:
    - `imu.global_mean/std`: length should match **30 dims** (5×6 flattened)
    - `airpods.global_mean/std`: recommended **6 dims** (only acc(3) + gyro(3) used)

### 3.2 Label mapping (`mapping_path`)

Requires `label_mapping.json`, example:

```json
{
  "id_to_action": {"0": "Stretching", "1": "Pouring Water"},
  "old_to_new_mapping": {"0": 0, "1": 1}
}
```

- `id_to_action`: old class ID -> name
- `old_to_new_mapping`: old class ID -> new class ID (used for merging)

### 3.3 Test set directory (`test_dataset_path` + `dataset_root_path`)

Testing runs inference on **file-level long sequences**:

- `test_dataset_path/test.csv`
  - Contains `file_name` column listing test file names
- `test_dataset_path/info.json`
- `test_dataset_path/{modality}_annotations.json`
  - GT for mAP evaluation (ActivityNet-style json)
- `dataset_root_path/<modality>/<file_name>`
  - Each `file_name` corresponds to one h5, read by `WWADLDatasetTestSingle`

---

## 4. Pretrained backbone (optional)

The repo provides a 1D CNN backbone (`pre_train/pre_model.py`) registered in `builder_pretrainbackbone.py` via `PRETRAINED_ZOO`:

- Update `PRETRAINED_ZOO["CNN1D"]["ckpt"]` to the correct path on your machine.
- If the ckpt is missing, training falls back to **random initialization + train_backbone=True** (see `train_epoch.py`).

> Note: `pre_train/pre_imu.py` references a missing `pre_tsse_mamba_model_7s`. To run that script, add the missing file or remove the import.

---

## 5. Training and testing

### 5.1 One-click run

**(XRFV2)**

Edit `base_config` in `run_main_xrfv2.py` (especially the `path` fields), then run:

```bash
python run_main_xrfv2.py
```

The script runs each experiment in `experiments = [...]`:

- WSDDN training -> save best ckpt
- Load ckpt for testing -> generate `predictions.json` / `train_test_report.txt`
- PCL/OICR follows the same pattern (e.g., `predictions_pcl.json`)

**(Other datasets)**

Edit `base_config` in `run_wsddn/pcl_{dataset_name}.py` (especially `path`), then run:

```bash
python run_wsddn/pcl_{dataset_name}.py
```

The script runs each experiment in `experiments = [...]`:

- WSDDN training -> save best ckpt
- Load ckpt for testing -> generate `predictions.json` / `train_test_report.txt`
- PCL/OICR follows the same pattern (e.g., `predictions_pcl.json`)

### 5.2 Output files

Controlled by `config["path"]["result_path"]`, typically:

- `inference_stats.json`: inference time / GPU peak memory stats
- `predictions.json` or `predictions_pcl.json`: ActivityNet-style prediction results
- `train_test_report.txt`: mAP@tIoU report

### 5.3 Multi-threading/CPU contention (PCL KMeans)

`sklearn.cluster.KMeans` runs on CPU. If you see thread contention or slowdowns, try:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

and reduce `num_workers` in code.

---

## 6. FAQ

1) **Pretrained ckpt not found**: update `PRETRAINED_ZOO` `ckpt` path in `builder_pretrainbackbone.py`.

2) **`global_stats.json` dimension mismatch**:
- IMU needs 30 dims (5×6 flattened)
- AirPods should be 6 dims (acc+gyro), matching `dataset_xrfv2.py::_preprocess_airpods()`

3) **Test set loading fails**: ensure `test_dataset_path/test.csv` exists and includes a `file_name` column; verify `dataset_root_path/<modality>/<file_name>` points to actual data files.

---
