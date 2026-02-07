# WeaklySupervised Sound (DCASE/CDur)

This repository provides **weakly supervised temporal localization** for IMU/audio-like sequence data with **two models only**:

- **DCASE (CRNN)**
- **CDur**

Supported datasets:
`xrfv2`, `hangtime`, `opportunity`, `rwhar`, `sbhar`, `wear`, `wetlab`.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Unified Entry

All training/testing and dry runs go through **`scripts/run.py`**.

```bash
python scripts/run.py --dataset xrfv2 --model dcase --config configs/xrfv2.json --dry_run
python scripts/run.py --dataset xrfv2 --model cdur  --config configs/xrfv2.json --dry_run
python scripts/run.py --dataset wear  --model dcase --config configs/wear.json  --dry_run
python scripts/run.py --dataset wear  --model cdur  --config configs/wear.json  --dry_run
```

Key arguments:

- `--dataset {xrfv2,hangtime,opportunity,rwhar,sbhar,wear,wetlab}`
- `--model {dcase,cdur}`
- `--mode {train,test}`
- `--dry_run` (no data required; uses dummy data)
- `--override key=value` (override JSON config fields)

## 3. Configuration

Each dataset has a JSON file in `configs/`. These define:

- `paths.data_dir`: dataset root (relative to repo by default)
- `data`: fps, clip length, input channels, class count
- `model`: DCASE/CDur hyperparameters
- `training`: batch size, lr, label smoothing, etc.

**Note:** `clip_frames` is tuned for dry-run defaults. For real training, set it to the actual frame length in your dataset.

## 4. Dataset Layout

- **XRFV2**: `dataset/dataset_xrfv2.py` / `dataset/dataset_XRFV2_loso.py`
- **Other datasets**: `OtherData/<DATASET>/dataset_*_ws.py`

## 5. Docs

- `docs/PROJECT_STRUCTURE.md`: final directory layout and execution flow
- `docs/CLEANUP_NOTES.md`: deletion/cleanup summary

## 6. Sanity Checks

```bash
python -m compileall .
```
