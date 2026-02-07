# Project Structure (WeaklySupervised Sound)

This repository has been slimmed to keep only **DCASE** and **CDur** models and their data pipelines across 7 datasets.

```
WeaklySupervised_Sound/
├── configs/
│   ├── xrfv2.json
│   ├── hangtime.json
│   ├── opportunity.json
│   ├── rwhar.json
│   ├── sbhar.json
│   ├── wear.json
│   └── wetlab.json
├── dataset/
│   ├── dataset_xrfv2.py
│   └── dataset_XRFV2_loso.py
├── OtherData/
│   ├── HANGTIME/
│   │   └── dataset_hangtime_ws.py
│   ├── Opportunity/
│   │   └── dataset_opportunity_ws.py
│   ├── RWHAR/
│   │   └── dataset_rwhar_ws.py
│   ├── SBHAR/
│   │   └── dataset_sbhar_ws.py
│   ├── WEAR/
│   │   └── dataset_wear_ws.py
│   └── WETLAB/
│       └── dataset_wetlab_ws.py
├── models/
│   ├── dcase.py
│   ├── cdur.py
│   ├── DCASE_CNN.py
│   └── DCASE_RNN.py
├── scripts/
│   └── run.py
├── utils/
│   ├── config.py
│   ├── dry_run.py
│   ├── losses.py
│   ├── postprocess.py
│   └── seed.py
├── docs/
│   ├── CLEANUP_NOTES.md
│   └── PROJECT_STRUCTURE.md
├── tool.py
└── README.md
```

## Unified Entry

- **`scripts/run.py`** is the only entry point.
- Supports:
  - `--dataset {xrfv2,hangtime,opportunity,rwhar,sbhar,wear,wetlab}`
  - `--model {dcase,cdur}`
  - `--config configs/<dataset>.json`
  - `--mode {train,test}`
  - `--dry_run` (no data required)

## Training/Testing Flow

1. `scripts/run.py` loads the JSON config via `utils.config.load_json_config`.
2. A model is created from `models/dcase.py` or `models/cdur.py`.
3. Dataset loaders are created from:
   - XRFV2: `dataset/dataset_xrfv2.py`
   - Other datasets: `OtherData/<DATASET>/dataset_*_ws.py`
4. Common training/evaluation logic uses `utils/losses.py` and `utils/postprocess.py`.
5. `--dry_run` builds a dummy dataset (`utils/dry_run.py`) and runs 1–2 steps.

## Dataset Entry Points

| Dataset | Module | Class |
|---------|--------|-------|
| xrfv2 | `dataset.dataset_xrfv2` | `WeaklySupervisedXRFV2DatasetTrain/Test` |
| hangtime | `OtherData.HANGTIME.dataset_hangtime_ws` | `WeaklyHangtimeDataset` |
| opportunity | `OtherData.Opportunity.dataset_opportunity_ws` | `WeaklyOpportunityDataset` |
| rwhar | `OtherData.RWHAR.dataset_rwhar_ws` | `WeaklyRWHARDataset` |
| sbhar | `OtherData.SBHAR.dataset_sbhar_ws` | `WeaklySBHARDataset` |
| wear | `OtherData.WEAR.dataset_wear_ws` | `WeaklyWearDataset` |
| wetlab | `OtherData.WETLAB.dataset_wetlab_ws` | `WeaklyWetlabDataset` |
