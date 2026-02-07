# Cleanup Notes

## Removed (WSDDN/PCL/OICR related)

- `models/WSDDN_model.py`
- `models/PCL_OICR_model.py`
- `models/PCLHead.py`
- `models/pcl_model_blocks.py`
- `builder_models.py`, `builder_pretrainbackbone.py`, `train_epoch.py`, `test_epoch.py`, `run_main_xrfv2.py`
- All `OtherData/*/run_wsddn_*.py` and `OtherData/*/run_pcl_*.py` scripts

## Removed (duplicated or legacy DCASE/CDur scripts)

- Root-level `run_dcase_xrfv2*.py`, `run_cdur_xrfv2*.py`, `*_onlytest.py`, `xrfv2_dcase_loso_test.py`
- Historical WEAR scripts (`run_dcase_wear_2022.py`, `run_dcase_wear_2026.py`)
- `postprocessing/` legacy evaluation scripts tied to old pipelines

## Model Consolidation

- DCASE implementations are unified in `models/dcase.py` (supports CRNN).
- CDur implementations are unified in `models/cdur.py`.

## Dataset Decoupling

- `dataset/dataset_xrfv2.py` and `dataset/dataset_XRFV2_loso.py` no longer import WSDDN.
- Proposal boxes (WSDDN-only) were removed from dataset outputs.

## Notes

- Pre-train scripts under `OtherData/*/pre_train` were removed because they were tied to the old WSDDN/PCL workflow.
- `tool.py` and `eval_xrfv2_metrics.py` remain for evaluation utilities.
