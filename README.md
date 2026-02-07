# WeaklySupervised Meta-Repository (Image / Sound / Video)

## 1) Repository Overview

This meta-repo contains three weakly supervised subprojects:

- **Image**: WSDDN / OICR / PCL
- **Sound**: DCASE / CDur
- **Video**: CoLA / RSKP

Datasets referenced across the subprojects (confirmed via configs and docs) include: `xrfv2`, `hangtime`, `opportunity`, `rwhar`, `sbhar`, `wear`, and `wetlab`.

## 2) Repository Layout

Detected true roots (entrypoints and configs live at these paths):

- `WeaklySupervised_Image/`
- `WeaklySupervised_Sound/`
- `WeaklySupervised_Video/`

Top-level tree:

```
WeaklySupervised/
├── WeaklySupervised_Image/
├── WeaklySupervised_Sound/
├── WeaklySupervised_Video/
├── requirements.txt
├── tools/
│   └── check_imports.py
└── README.md
```

## 3) Installation

Recommended Python version: **Python 3.8+**.

```bash
# From the meta-repo root
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: choose the appropriate PyTorch CUDA/CPU build for your system. The requirements file does not pin CUDA wheels.

## 4) Quick Start (Dry Run / Smoke Test)

All commands below are real entrypoints in each subproject. Run them from the stated working directory.

### Image (WSDDN / OICR / PCL)

```bash
cd WeaklySupervised_Image
python scripts/run.py --dataset xrfv2 --model wsddn --dry_run
python scripts/run.py --dataset xrfv2 --model pcl --dry_run
```

### Sound (DCASE / CDur)

```bash
cd WeaklySupervised_Sound
python scripts/run.py --dataset xrfv2 --model dcase --config configs/xrfv2.json --dry_run
python scripts/run.py --dataset xrfv2 --model cdur  --config configs/xrfv2.json --dry_run
```

### Video (CoLA / RSKP)

```bash
cd WeaklySupervised_Video
python scripts/run.py --dataset xrfv2 --model rskp --dry_run
python scripts/run.py --dataset xrfv2 --model cola --dry_run
```

You can also verify that the third-party dependencies listed in `requirements.txt` import correctly:

```bash
python tools/check_imports.py
```

## 5) Subproject Notes

- Each subproject keeps dataset/model configuration in its `configs/` directory (JSON).
- Use `--dry_run` first to validate the pipeline before running with real data.
- For deeper details, see each subproject's README and docs under its directory.
