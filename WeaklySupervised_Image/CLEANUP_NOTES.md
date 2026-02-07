# Cleanup Notes

## Removed items

* `RSKP_MODEL/` (entire directory)
* `run_main_xrfv2_test_rskp.py`
* `OtherData/*/run_rskp_*.py`
  * `OtherData/HANGTIME/run_rskp_hangtime.py`
  * `OtherData/Opportunity/run_rskp_opportunity.py`
  * `OtherData/RWHAR/run_rskp_rwhar.py`
  * `OtherData/SBHAR/run_rskp_sbhar.py`
  * `OtherData/WEAR/run_rskp_wear.py`
  * `OtherData/WETLAB/run_rskp_wetlab.py`
* `calculate_model_params.py`
* `__pycache__/` directories and `*.pyc` artifacts

## Retained (by design)

* Pretraining helpers under `pre_train/` and `OtherData/*/pre_train/` remain for optional backbone training.
* Visualization/util scripts (e.g., `plot_timeline.py`) were kept but normalized to relative paths.

No other uncertain files were retained.
