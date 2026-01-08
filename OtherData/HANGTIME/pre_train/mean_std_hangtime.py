# dataset_hangtime.py  (NPY-only + LOSO mean/var cached as JSON under raw/)
import os
import json
import re
import math
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# 1) LOSO json parsing
# ============================================================
def _load_loso_json(json_path: str):
    """
    读取 loso_sbj_k.json
    输出:
      loso_db[sbj] = {"subset": "Training"/"Validation", "fps": 50, "annos": [(s,e,lid), ...]}
    其中 s,e 是 frame index（int），区间按 [s, e) 使用。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    db = obj["database"]

    parsed = {}
    for sbj, info in db.items():
        subset = info.get("subset", None)
        fps = int(info.get("fps", 50))  # ★RWHAR 默认 50Hz
        annos = []
        for a in info.get("annotations", []):
            seg_f = a.get("segment (frames)", None)
            if not seg_f:
                continue
            s = int(round(float(seg_f[0])))
            e = int(round(float(seg_f[1])))
            lid = int(a["label_id"])
            if e > s:
                annos.append((s, e, lid))
        annos.sort(key=lambda x: x[0])
        parsed[sbj] = {"subset": subset, "fps": fps, "annos": annos}
    return parsed


def _subjects_by_subset(loso_db: dict, subset_name: str):
    subs = [k for k, v in loso_db.items() if v.get("subset") == subset_name]
    subs.sort()
    return subs


def _parse_fold_id_from_loso_name(loso_json: str):
    base = os.path.basename(loso_json)
    m = re.search(r"loso_sbj_(\d+)", base)
    if m:
        return int(m.group(1))
    return None


# ============================================================
# 2) NPY -> continuous raw frames (50Hz)
# ============================================================
def npy_windows_to_raw_frames(
    npy_path: str,
    num_sensors: int = 21,     # ★RWHAR: 3轴*7设备=21
    win_samples: int = 50,     # ★RWHAR: 50_samples
    overlap: float = 0.5,      # 50% overlap
    dtype=np.float32,
):
    """
    RWHAR npy: shape [N, num_sensors*win_samples]，例如 [8632, 1050] = 21*50
    flatten 顺序为 sensor-major:
      sensor0(win) + sensor1(win) + ... + sensor20(win)

    overlap=0.5 -> stride = win_samples*(1-0.5)=25 samples

    还原连续 raw：
      第一个窗口取完整 win_samples，
      后续窗口只追加“新出现的那部分帧”（窗口后半段 stride 部分）。
    返回:
      raw_frames: [T_frames, num_sensors]
    """
    feat = np.load(npy_path, allow_pickle=False).astype(dtype)  # [N, 1050]
    if feat.ndim != 2:
        raise ValueError(f"expect 2D npy, got {feat.shape} from {npy_path}")
    if feat.shape[1] != num_sensors * win_samples:
        raise ValueError(
            f"dim mismatch in {npy_path}: {feat.shape[1]} != {num_sensors}*{win_samples}"
        )

    step = int(round(win_samples * (1.0 - overlap)))  # 50*(1-0.5)=25
    if step <= 0 or step > win_samples:
        raise ValueError(f"bad overlap={overlap} for win_samples={win_samples}")
    ov = win_samples - step  # overlap samples, here 25

    # [N, 1050] -> [N, num_sensors, win] -> [N, win, num_sensors]
    x = feat.reshape(-1, num_sensors, win_samples).transpose(0, 2, 1)  # [N, 50, 21]

    out = [x[0]]  # first window full
    if x.shape[0] > 1:
        out.append(x[1:, ov:, :].reshape(-1, num_sensors))  # append only new frames
    raw = np.concatenate(out, axis=0)  # [T_frames, 21]
    return raw


# ============================================================
# 3) LOSO stats cache as JSON under raw/
# ============================================================
def _stats_dir_in_raw(dataset_dir: str, stats_dirname: str = "loso_norm_stats_json"):
    return os.path.join(dataset_dir, "raw", stats_dirname)


def _stats_json_path(dataset_dir: str, fold_id: int, stats_dirname: str = "loso_norm_stats_json"):
    d = _stats_dir_in_raw(dataset_dir, stats_dirname)
    return os.path.join(d, f"loso_sbj_{fold_id}_stats.json")


def _json_dump(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _json_load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_mean_var_across_subjects(
    dataset_dir: str,
    subjects: list,
    num_sensors: int,
    npy_rel_dir: str,
    win_samples: int,
    overlap: float,
    ignore_zeros_in_stats: bool,
):
    """
    基于还原后的连续 raw 序列（50Hz）统计每通道 mean/var。
    """
    sum_ = np.zeros((num_sensors,), dtype=np.float64)
    sumsq = np.zeros((num_sensors,), dtype=np.float64)
    count = np.zeros((num_sensors,), dtype=np.float64)

    npy_dir = os.path.join(dataset_dir, npy_rel_dir)

    for sbj in subjects:
        npy_path = os.path.join(npy_dir, f"{sbj}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"NPY not found: {npy_path}")

        raw = npy_windows_to_raw_frames(
            npy_path=npy_path,
            num_sensors=num_sensors,
            win_samples=win_samples,
            overlap=overlap,
            dtype=np.float32,
        )  # [T,C]
        x = raw.astype(np.float64)

        if not ignore_zeros_in_stats:
            sum_ += x.sum(axis=0)
            sumsq += (x * x).sum(axis=0)
            count += x.shape[0]
        else:
            m = (x != 0.0)
            count += m.sum(axis=0)
            x_nz = np.where(m, x, 0.0)
            sum_ += x_nz.sum(axis=0)
            sumsq += (x_nz * x_nz).sum(axis=0)

    safe_count = np.maximum(count, 1.0)
    mean = sum_ / safe_count
    var = (sumsq / safe_count) - (mean * mean)
    var = np.maximum(var, 0.0)

    for c in range(num_sensors):
        if count[c] < 2:
            mean[c] = 0.0
            var[c] = 0.0

    return mean.astype(np.float32), var.astype(np.float32), count.astype(np.float32)


def ensure_all_loso_stats_json(
    dataset_dir: str,
    folds=tuple(range(15)),  # ★RWHAR: 15 folds
    annotations_dirname="annotations",
    stats_dirname="loso_norm_stats_json",
    fps: int = 50,           # ★RWHAR: 50Hz
    num_sensors: int = 21,   # ★RWHAR: 3*7=21
    npy_rel_dir: str = os.path.join("processed", "inertial_features", "50_samples_50_overlap"),
    win_samples: int = 50,   # ★RWHAR: 50_samples
    overlap: float = 0.5,
    ignore_zeros_in_stats: bool = False,
):
    """
    确保每个 fold 的 stats 都存在：
      raw/loso_norm_stats_json/loso_sbj_k_stats.json
    """
    for k in folds:
        stats_path = _stats_json_path(dataset_dir, k, stats_dirname)
        if os.path.exists(stats_path):
            continue

        loso_json_path = os.path.join(dataset_dir, annotations_dirname, f"loso_sbj_{k}.json")
        if not os.path.exists(loso_json_path):
            raise FileNotFoundError(f"LOSO json not found: {loso_json_path}")

        loso_db = _load_loso_json(loso_json_path)
        train_subjects = _subjects_by_subset(loso_db, "Training")
        if len(train_subjects) == 0:
            raise RuntimeError(f"Fold {k}: no Training subjects found in {loso_json_path}")

        mean, var, count = _compute_mean_var_across_subjects(
            dataset_dir=dataset_dir,
            subjects=train_subjects,
            num_sensors=num_sensors,
            npy_rel_dir=npy_rel_dir,
            win_samples=win_samples,
            overlap=overlap,
            ignore_zeros_in_stats=ignore_zeros_in_stats,
        )

        payload = {
            "src": "npy",
            "fold_id": int(k),
            "fps": int(fps),
            "num_sensors": int(num_sensors),
            "npy_rel_dir": str(npy_rel_dir).replace("\\", "/"),
            "win_samples": int(win_samples),
            "overlap": float(overlap),
            "ignore_zeros_in_stats": bool(ignore_zeros_in_stats),
            "train_subjects": train_subjects,
            "mean": mean.tolist(),
            "var": var.tolist(),
            "count_per_channel": count.tolist(),
        }
        _json_dump(stats_path, payload)


def load_fold_mean_std_from_json(
    dataset_dir: str,
    fold_id: int,
    stats_dirname: str,
    eps: float,
    expect: dict,
):
    stats_path = _stats_json_path(dataset_dir, fold_id, stats_dirname)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats json missing: {stats_path}")

    obj = _json_load(stats_path)

    for k, v in expect.items():
        if k not in obj:
            raise RuntimeError(f"Stats json missing key='{k}': {stats_path}")
        if obj[k] != v:
            raise RuntimeError(
                f"Stats json meta mismatch for '{k}': file={obj[k]} vs expect={v} | {stats_path}"
            )

    mean = np.asarray(obj["mean"], dtype=np.float32)
    var = np.asarray(obj["var"], dtype=np.float32)
    std = np.sqrt(var + float(eps)).astype(np.float32)
    return mean, std


if __name__ == "__main__":
    dataset_dir = "/data/bx_data/TAL_data/hangtime/"
    ensure_all_loso_stats_json(
        dataset_dir=dataset_dir,
        folds=tuple(range(24)),
        stats_dirname="loso_norm_stats_json",
        fps=50,
        num_sensors=3,
        npy_rel_dir=os.path.join("processed", "inertial_features", "50_samples_50_overlap"),
        win_samples=50,
        overlap=0.5,
        ignore_zeros_in_stats=False,
    )
    print("Done. Stats saved under:", _stats_dir_in_raw(dataset_dir, "loso_norm_stats_json"))
