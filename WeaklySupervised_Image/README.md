# Weakly Supervised IMU (Image) Benchmarks

This repository keeps only the three weakly-supervised detectors (WSDDN / OICR / PCL) and seven datasets (XRFV2 + six OtherData datasets). It includes a unified runner, dataset-specific configs, and a dry-run mode that exercises model forward passes without real data.

## Supported models

* **WSDDN family** (`models/WSDDN_model.py`)
  * `wsddn`
  * `wsddn_avg`
  * `wsddn_transformer`
* **PCL / OICR IMU** (`models/PCL_OICR_model.py`)
  * `pcl_imu`
  * `oicr_imu`
* **OICR BUAA variant** (`models/oicr_paloss_model.py`) used by the `run_oicrBUAA_*` scripts under `OtherData/`

Backbone checkpoints are configured via JSON using `model.ckpt_root` or `model.ckpt_path`. If the checkpoint is missing, training will fall back to random initialization.

## Supported datasets

* `xrfv2`
* `hangtime`
* `opportunity`
* `rwhar`
* `sbhar`
* `wear`
* `wetlab`

## Unified runner

Use `scripts/run.py` to launch training/testing or dry-run execution.

```bash
# Dry-run (dummy data, no real files needed)
python scripts/run.py --dataset xrfv2 --model wsddn --dry_run
python scripts/run.py --dataset xrfv2 --model pcl --dry_run
python scripts/run.py --dataset xrfv2 --model oicr --dry_run
python scripts/run.py --dataset wear --model wsddn --dry_run
python scripts/run.py --dataset wear --model pcl --dry_run
python scripts/run.py --dataset wear --model oicr --dry_run

# Real training/testing (edit configs first)
python scripts/run.py --dataset xrfv2 --model wsddn --config configs/xrfv2_wsddn.json
python scripts/run.py --dataset wear --model oicr --config configs/wear_oicr.json
```

## Configuration files

All training/evaluation parameters live under `configs/`. Each dataset-model pair has a JSON file (e.g. `configs/opportunity_pcl.json`). Adjust these paths for your environment:

* **XRFV2** expects:
  * `path.mapping_path`
  * `path.train_dataset_path/global_stats.json`
  * `path.test_dataset_path/info.json`
  * `path.test_dataset_path/imu_annotations.json`
* **OtherData datasets** expect dataset roots, normalization stats, and (optionally) pre-trained backbones in the corresponding `OtherData/<DATASET>/pre_train` directory.

If a required file is missing, training will fail fast with a clear error.

## Legacy entry scripts

Dataset-specific entrypoints remain available under `OtherData/*/run_*.py` but now accept `--config` JSON paths. Example:

```bash
python OtherData/WEAR/run_wsddn_wear.py --config configs/wear_wsddn.json
```

## Project structure

See `PROJECT_STRUCTURE.md` for a detailed directory map and entrypoint references.

## Cleanup notes

See `CLEANUP_NOTES.md` for the removal log and retained files.
