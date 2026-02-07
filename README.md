# WeaklySupervised

This repository combines three subprojects from the paper into a single, reproducible codebase. The code is organized by modality and model family, and each subproject can be run independently with the shared dataset layout for the 7 IMU datasets.

## Project layout

- `CoLA/` — video-domain CoLA model experiments on 7 IMU datasets.
- `sound/` — audio-domain CDur and DCASE models on 7 IMU datasets.
- `WSDDN/` — image-domain WSDDN/OICR/PCL models and the video-domain RSKP model on 7 IMU datasets.

Each subproject contains its own run scripts, dataset loaders, and evaluation utilities. This top-level README and the unified `requirements.txt` provide a single entry point for setup and reproduction.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running experiments

Each subproject exposes dedicated run scripts. Examples:

### CoLA (video)

```bash
python CoLA/run_cola_xrfv2_ddp.py
```

### CDur / DCASE (audio)

```bash
python sound/run_cdur_xrfv2.py
python sound/run_dcase_xrfv2.py
```

### WSDDN / OICR / PCL / RSKP (image + video)

```bash
python WSDDN/run_main_xrfv2.py
python WSDDN/run_main_xrfv2_test_rskp.py
```

Refer to each subproject's local README and run scripts for dataset paths, dataset splits, and configuration options.

## Notes

- The codebase was consolidated without altering model logic.
- Chinese code comments have been removed for consistency.
- Keep dataset layouts consistent with the original scripts.
