# WeaklySupervisedTemporalActionLocalization (IMU)

这个仓库是一个 **弱监督时序动作定位（Weakly-Supervised Temporal Action Localization, WS-TAL）** 的研究/实验代码，主要面向 **可穿戴/IMU 等 1D 传感器序列**（支持 IMU + 可选 AirPods 传感器拼接）。

当前主干实现：

- **WSDDN-IMU**：WSDDN 双分支（classification + detection）用于弱监督定位，proposal pooling 使用 **1D Temporal SPP**（`TemporalSPP1D`）。
- **PCL/OICR-IMU**：在 WSDDN/MIL 之上加入 **OICR refine**，并可选 **PCL (Proposal Cluster Learning)**，内部用 **KMeans** 生成 pseudo GT / cluster centers。
- **推理与评估**：按滑窗对长序列推理，按类做 **Soft-NMS**，用 ActivityNet 风格的 `ANETdetection` 计算 **mAP@tIoU**。

---

## 1. 环境与安装
```bash
pip install -r requirements.txt
```
---

## 2. 代码结构（核心文件）
项目分两部分：一部分是xrfv2数据集，另一部分是其他数据集。xrfv2数据集的训练和测试代码在主文件夹下，其他数据集的训练和测试在OtherData文件夹下。（其他数据集的结构都是一样的）
dataset：xrfv2的在./dataset/dataset_xrfv2.py  ;  其他数据集在各自目录下的dataset_{数据集名称}_ws.py

-`run_main_xrfv2.py`：**XRFV2** 主入口（在脚本中直接写 config，并串起训练+测试）。
- `train_epoch.py`：训练循环
  - `train_wsddn_imu(config)`
  - `train_pcl_imu(config)`
- `test_epoch.py`：测试/推理与评估
  - `test_wsddn_imu(config, checkpoint_path)`
  - `test_pcl_imu(config, checkpoint_path)`
- `dataset/dataset_xrfv2.py`：XRFV2/WWADL 风格数据集（训练用 `*.h5`，测试用 `test.csv` + per-file h5）
- `models/WSDDN_model.py`：WSDDN + TemporalSPP1D + proposal 生成等
- `models/PCL_OICR_model.py`：PCL/OICR 头（KMeans + refine）
- `builder_pretrainbackbone.py`：预训练 backbone 的注册与加载（`PRETRAINED_ZOO`）
- `tool.py`：Soft-NMS、ANETdetection(mAP) 等工具

### 2.1 其他数据集

`OtherData/` 下提供了多个数据集的弱监督训练/测试脚本（例如 `Opportunity/`、`RWHAR/`、`HANGTIME/`、`WEAR/`、`WETLAB/`、`SBHAR/`），整体流程与根目录类似，但各自的数据组织与 LOSO 划分略有不同。

---

## 3. 数据准备（以 XRFV2 为例）

### 3.1 训练集目录（`train_dataset_path`）

训练集通过 30s clip 级弱监督进行训练，默认期望：

- `train_data.h5`
  - `imu`：形状 **[N, 2048, 5, 6]**（2048 为 30s 的采样点数；5 个设备；每设备 6 维）
  - `airpods`（可选）：形状 **[N, 2048, 9]**（实际使用 AirPods 的 acc(3) + gyro(3)）
- `train_label.json`
  - JSON 内包含 `imu` 字段：`{sample_idx(str): [ [left_offset, right_offset, old_label_id], ... ] }`
  - `left_offset/right_offset` 为相对位置
- `global_stats.json`
  - 包含每个模态的全局均值与方差：
    - `imu.global_mean/std`：长度应匹配 **30 维**（5×6 展平）
    - `airpods.global_mean/std`：建议直接保存为 **6 维**（代码只使用 AirPods 的 acc(3) + gyro(3)）

### 3.2 Label 映射（`mapping_path`）

需要 `label_mapping.json`，示例字段：

```json
{
  "id_to_action": {"0": "Stretching", "1": "Pouring Water"},
  "old_to_new_mapping": {"0": 0, "1": 1}
}
```

- `id_to_action`：旧类别 ID -> 名称
- `old_to_new_mapping`：旧类别 ID -> 新类别 ID（可用于合并类别）

### 3.3 测试集目录（`test_dataset_path` + `dataset_root_path`）

测试阶段按 **文件级长序列** 推理：

- `test_dataset_path/test.csv`
  - 包含 `file_name` 列，用于列出要测试的文件名
- `test_dataset_path/info.json`
- `test_dataset_path/{modality}_annotations.json`
  - 用于 mAP 评估的 GT（ActivityNet 风格 json）
- `dataset_root_path/<modality>/<file_name>`
  - 每个 `file_name` 对应一个 h5，供 `WWADLDatasetTestSingle` 读取

---

## 4. 预训练 backbone（可选）

仓库提供了 1D CNN backbone（`pre_train/pre_model.py`），并在 `builder_pretrainbackbone.py` 里通过 `PRETRAINED_ZOO` 注册：

- 需要把 `PRETRAINED_ZOO["CNN1D"]["ckpt"]` 改成你机器上的权重路径。
- 若找不到 ckpt，训练阶段会自动走 **随机初始化 + train_backbone=True**（见 `train_epoch.py`）。

> 注意：`pre_train/pre_imu.py` 中引用了未包含的 `pre_tsse_mamba_model_7s`，如果你要运行该脚本，需要补齐对应文件或删掉相关 import。

---

## 5. 训练与测试

### 5.1 一键运行
（XRFV2）

直接修改 `run_main_xrfv2.py` 里的 `base_config`（尤其是 `path` 相关字段），然后运行：
```bash
python run_main_xrfv2.py
```

脚本会按 `experiments = [...]` 逐个实验执行：

- WSDDN 训练 → 保存最优 ckpt
- 加载 ckpt 进行测试 → 生成 `predictions.json` / `train_test_report.txt`
- PCL/OICR 同理（输出 `predictions_pcl.json` 等）

（其他数据集）

直接修改 `run_wsddn/pcl_{对应数据集名}.py` 里的 `base_config`（尤其是 `path` 相关字段），然后运行：
```bash
python run_wsddn/pcl_{对应数据集名}.py
```

脚本会按 `experiments = [...]` 逐个实验执行：

- WSDDN 训练 → 保存最优 ckpt
- 加载 ckpt 进行测试 → 生成 `predictions.json` / `train_test_report.txt`
- PCL/OICR 同理（输出 `predictions_pcl.json` 等）

### 5.2 输出文件

由 `config["path"]["result_path"]` 控制，典型会生成：

- `inference_stats.json`：推理耗时 / GPU 峰值显存统计
- `predictions.json` 或 `predictions_pcl.json`：ActivityNet 风格预测结果
- `train_test_report.txt`：mAP@tIoU 报告

### 5.3 多线程/CPU 竞争（PCL 的 KMeans）

PCL 中的 `sklearn.cluster.KMeans` 在 CPU 上运行，若遇到线程打架/速度异常，可尝试：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

并在代码里减少 `num_workers`。

---

## 6. 常见问题（FAQ）

1) **找不到预训练 ckpt**：请修改 `builder_pretrainbackbone.py` 中 `PRETRAINED_ZOO` 的 `ckpt` 路径。

2) **`global_stats.json` 维度不匹配**：
- IMU 需要 30 维（5×6 展平）
- AirPods 建议保存为 6 维（acc+gyro），与 `dataset_xrfv2.py::_preprocess_airpods()` 对齐

3) **测试集读取失败**：确认 `test_dataset_path/test.csv` 存在且包含 `file_name` 列；并确保 `dataset_root_path/<modality>/<file_name>` 指向实际数据文件。

---

