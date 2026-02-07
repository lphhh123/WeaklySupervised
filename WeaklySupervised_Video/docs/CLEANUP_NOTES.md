# Cleanup Notes

This refactor focuses on the two supported models (CoLA + RSKP) and the seven supported datasets (XRFV2 + six OtherData datasets).

## Kept legacy files

The following legacy files remain in the repository because they may still be referenced by downstream scripts or for historical experiments:

- `CoLA/eval/` and `eval_xrfv2_metrics.py`: legacy evaluation utilities.
- `pre_train/` and `OtherData/**/pre_train/`: pretraining helpers (paths still need project-specific data/weights).
- `OtherData/utils.py` and `tool.py`: shared evaluation/utility helpers reused by dataset wrappers.
- `CoLA/pre_models/` (also mirrored into `models/cola/pre_models/`): legacy backbone implementations, some of which still include absolute-path sys.path references. These are untouched but now optional.

If you want a slimmer repo, you can delete these once you confirm you no longer need them.

## Removed items

- `__pycache__/` directories and `.pyc` files were deleted as part of cleanup.

## Notes on paths

Hardcoded `/home/...` paths have been removed or moved to JSON config defaults where possible. Any remaining absolute paths are confined to legacy scripts under `pre_train/` or `CoLA/eval/`, and are not used by the unified entrypoint.
