# Project Structure

## Key directory tree (trimmed)

```
WeaklySupervised_Image/
├── builder_models.py
├── builder_pretrainbackbone.py
├── configs/
│   ├── xrfv2_wsddn.json
│   ├── xrfv2_pcl.json
│   ├── xrfv2_oicr.json
│   ├── hangtime_wsddn.json
│   ├── hangtime_pcl.json
│   ├── hangtime_oicr.json
│   ├── opportunity_wsddn.json
│   ├── opportunity_pcl.json
│   ├── opportunity_oicr.json
│   ├── rwhar_wsddn.json
│   ├── rwhar_pcl.json
│   ├── rwhar_oicr.json
│   ├── sbhar_wsddn.json
│   ├── sbhar_pcl.json
│   ├── sbhar_oicr.json
│   ├── wear_wsddn.json
│   ├── wear_pcl.json
│   ├── wear_oicr.json
│   ├── wetlab_wsddn.json
│   ├── wetlab_pcl.json
│   └── wetlab_oicr.json
├── dataset/
│   └── dataset_xrfv2.py
├── models/
│   ├── WSDDN_model.py
│   ├── PCL_OICR_model.py
│   ├── pcl_model_blocks.py
│   ├── PCLHead.py
│   └── oicr_paloss_model.py
├── OtherData/
│   ├── HANGTIME/
│   │   ├── dataset_hangtime_ws.py
│   │   ├── run_wsddn_hangtime.py
│   │   ├── run_pcl_hangtime.py
│   │   └── run_oicrBUAA_hangtime.py
│   ├── Opportunity/
│   │   ├── dataset_opportunity_ws.py
│   │   ├── run_wsddn_opportunity.py
│   │   ├── run_pcl_opportunity.py
│   │   └── run_oicrBUAA_opportunity.py
│   ├── RWHAR/
│   │   ├── dataset_rwhar_ws.py
│   │   ├── run_wsddn_rwhar.py
│   │   ├── run_pcl_rwhar.py
│   │   └── run_oicrBUAA_rwhar.py
│   ├── SBHAR/
│   │   ├── dataset_sbhar_ws.py
│   │   ├── run_wsddn_sbhar.py
│   │   ├── run_pcl_sbhar.py
│   │   └── run_oicrBUAA_sbhar.py
│   ├── WEAR/
│   │   ├── dataset_wear_ws.py
│   │   ├── run_wsddn_wear.py
│   │   ├── run_pcl_wear.py
│   │   └── run_oicrBUAA_wear.py
│   └── WETLAB/
│       ├── dataset_wetlab_ws.py
│       ├── run_wsddn_wetlab.py
│       ├── run_pcl_wetlab.py
│       └── run_oicrBUAA_wetlab.py
├── scripts/
│   ├── config_utils.py
│   └── run.py
├── train_epoch.py
├── test_epoch.py
└── run_main_xrfv2.py
```

## Model implementations

* **WSDDN family**: `models/WSDDN_model.py`
* **PCL / OICR IMU**: `models/PCL_OICR_model.py`, `models/pcl_model_blocks.py`, `models/PCLHead.py`
* **OICR BUAA variant**: `models/oicr_paloss_model.py`

## Datasets and entrypoints

### XRFV2

* Dataset: `dataset/dataset_xrfv2.py`
* Entrypoint: `run_main_xrfv2.py` (uses `train_epoch.py` and `test_epoch.py`)
* Unified runner config: `configs/xrfv2_*.json`

### OtherData (6 datasets)

Each dataset has a dataset module and three run scripts:

* `run_wsddn_*.py` (WSDDN)
* `run_pcl_*.py` (PCL/OICR)
* `run_oicrBUAA_*.py` (OICR BUAA)

### Unified entrypoint

* `scripts/run.py` accepts `--dataset` and `--model` and loads the JSON config files in `configs/`.

## Training / testing logic

* **XRFV2 WSDDN/PCL/OICR**
  * Training: `train_epoch.py` → `train_wsddn_imu`, `train_pcl_imu`
  * Testing: `test_epoch.py` → `test_wsddn_imu`, `test_pcl_imu`

* **OtherData datasets**
  * Training/Testing functions are defined within each `run_*` script and invoked via `run_entry()` wrappers.
