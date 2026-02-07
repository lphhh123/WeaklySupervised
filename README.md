# WeaklySupervised

This repository consolidates the original three codebases into a single, organized project for weakly supervised temporal action localization on 7 IMU datasets. The code remains unchanged (only documentation/comment cleanup), and the top-level structure is organized by modality and model family.

## Project Structure

```
./CoLA/   # Video-domain model (CoLA) experiments on 7 IMU datasets
./sound/  # Audio-domain models (CDur, DCASE) experiments on 7 IMU datasets
./WSDDN/  # Image-domain models (WSDDN, OICR, PCL) + video-domain model (RSKP)
```

### Modality and Model Mapping

| Folder | Modality | Models | Notes |
| --- | --- | --- | --- |
| `CoLA/` | Video | CoLA | Video-domain model experiments across 7 IMU datasets. |
| `sound/` | Sound | CDur, DCASE | Audio-domain models adapted for IMU experiments. |
| `WSDDN/` | Image + Video | WSDDN, OICR, PCL, RSKP | Image-domain weakly supervised models + video-domain RSKP. |

### Datasets

Across the three folders, the experiments target the same 7 IMU datasets:

- HANGTIME
- OPPORTUNITY
- RWHAR
- SBHAR
- WEAR
- WETLAB
- XRFV2

## Getting Started

### 1) Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run Experiments

Each subproject keeps its original entry points. Use the scripts in each folder as the source of truth:

- **CoLA:** `CoLA/run_cola_*_ddp.py` and `CoLA/main_cola.py`
- **sound:** `sound/run_*` (e.g., `run_cdur_*`, `run_dcase_*`)
- **WSDDN:** `WSDDN/run_main_xrfv2.py`, `WSDDN/OtherData/*/run_*`

Refer to the subfolder README files (if present) and the run scripts for dataset paths and experiment configuration.

## Notes

- The code is preserved as-is; only Chinese comments were removed to unify documentation style.
- This top-level README and `requirements.txt` provide a single entry point for reproducibility.
