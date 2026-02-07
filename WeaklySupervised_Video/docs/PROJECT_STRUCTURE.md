# Project Structure

This repository is organized around two models (CoLA + RSKP) and seven datasets (XRFV2 + six OtherData datasets). The unified entrypoint is `scripts/run.py`, and dataset/model defaults live in `configs/`.

## Key directories

```
WeaklySupervised_Video/
├── configs/                 # JSON configs per dataset
├── datasets/
│   ├── xrfv2.py              # XRFV2 dataset wrappers
│   └── otherdata/            # OtherData dataset wrappers (hangtime, wear, etc.)
├── models/
│   ├── cola/                 # CoLA model package
│   └── rskp/                 # RSKP model package
├── scripts/
│   └── run.py                # Unified entrypoint (train/test/dry_run)
├── utils/                    # config/seed/io/postprocess helpers
├── tool.py                   # legacy evaluation/utility helpers
├── eval_xrfv2_metrics.py     # legacy eval for XRFV2
└── OtherData/                # dataset assets + raw scripts (wrappers now point to scripts/run.py)
```

## Unified entrypoint

All datasets and models are run through the same interface:

```bash
python scripts/run.py --dataset xrfv2 --model rskp --dry_run
python scripts/run.py --dataset xrfv2 --model cola --dry_run
python scripts/run.py --dataset wear --model rskp --dry_run
python scripts/run.py --dataset wear --model cola --dry_run
```

### CLI options

- `--dataset`: `xrfv2|hangtime|opportunity|rwhar|sbhar|wear|wetlab`
- `--model`: `cola|rskp`
- `--mode`: `train|test` (currently used for logging only)
- `--config`: JSON config path (defaults to `configs/<dataset>.json`)
- `--dry_run`: build model + dummy dataloader + forward 1-2 steps
- `--device`: `cpu|cuda`
- `--seed`: random seed

## Dataset-specific wrappers

Legacy scripts are retained as thin wrappers, for example:

- `CoLA/run_cola_wear_ddp.py` → `scripts/run.py --dataset wear --model cola`
- `OtherData/WEAR/run_rskp_wear.py` → `scripts/run.py --dataset wear --model rskp`
- `run_main_xrfv2_test_rskp.py` → `scripts/run.py --dataset xrfv2 --model rskp`

These wrappers no longer contain hardcoded imports or absolute paths.
