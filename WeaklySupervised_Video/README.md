# WeaklySupervised Video (CoLA + RSKP)

This repo provides **two models** (CoLA + RSKP) and **seven datasets** (XRFV2 + six OtherData datasets) under a unified structure. The primary entrypoint is `scripts/run.py`, and dataset defaults live in `configs/` (JSON).

## Project layout (summary)

```
WeaklySupervised_Video/
├── configs/                 # JSON configs per dataset
├── datasets/                # Dataset wrappers (xrfv2 + otherdata/*)
├── models/                  # cola/ + rskp/ model packages
├── scripts/                 # unified entrypoint
├── utils/                   # config/seed/io/postprocess
├── tool.py                  # legacy utilities
├── eval_xrfv2_metrics.py    # legacy eval
└── docs/                    # project structure + cleanup notes
```

For more details, see `docs/PROJECT_STRUCTURE.md` and `docs/CLEANUP_NOTES.md`.

## Unified entrypoint

```bash
python scripts/run.py --dataset xrfv2 --model rskp --dry_run
python scripts/run.py --dataset xrfv2 --model cola --dry_run
python scripts/run.py --dataset wear --model rskp --dry_run
python scripts/run.py --dataset wear --model cola --dry_run
```

### CLI

- `--dataset`: `xrfv2|hangtime|opportunity|rwhar|sbhar|wear|wetlab`
- `--model`: `cola|rskp`
- `--mode`: `train|test` (currently informational)
- `--config`: JSON config path (defaults to `configs/<dataset>.json`)
- `--dry_run`: build model + dummy dataloader + forward 1-2 steps
- `--device`: `cpu|cuda`
- `--seed`: random seed

## Configs

All paths are relative by default:

- `paths.data_dir`: dataset root directory
- `paths.mapping_path`: label mapping JSON
- `paths.output_dir`: output directory
- `paths.ckpt_dir`: checkpoint directory

You can override config values by editing the JSON or passing `--config` to point at another file.

## Legacy wrappers

Old scripts are retained as thin wrappers that forward to `scripts/run.py`:

- `CoLA/run_cola_*_ddp.py`
- `OtherData/*/run_rskp_*.py`
- `run_main_xrfv2_test_rskp.py`

They no longer contain hardcoded paths or broken imports.
