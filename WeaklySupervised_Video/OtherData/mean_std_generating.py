# dataset_opportunity.py  (NPY-only + LOSO mean/var cached as JSON under raw/)
import os
import json
import re
import numpy as np


# ============================================================
# 1) LOSO json parsing
# ============================================================
def _load_loso_json(json_path: str):
    """
    Message loso_sbj_k.json
    Message:
      loso_db[sbj] = {"subset": "Training"/"Validation", "fps": 30, "annos": [(s,e,lid), ...]}
    Message s,e Message frame index（int），Message [s, e) Message。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    db = obj["database"]

    parsed = {}
    for sbj, info in db.items():
        subset = info.get("subset", None)
        fps = int(info.get("fps", 30))
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


# ============================================================
# 2) NPY -> continuous raw frames (30Hz)
# ============================================================
def npy_windows_to_raw_frames(
    npy_path: str,
    num_sensors: int = 113,
    win_samples: int = 30,
    overlap: float = 0.5,
    dtype=np.float32,
):
    """
    Message npy: shape [N, num_sensors*win_samples]，flatten Message sensor-major:
      sensor0(30) + sensor1(30) + ... + sensor112(30)

    npy Message: win_samples(=30) samples，overlap(=0.5) -> stride=15 samples

    Message 30Hz raw：
      Message 30 Message，
      Message“Message”（Message 15 Message）。

    Message:
      raw_frames: [T_frames, num_sensors]
    """
    feat = np.load(npy_path, allow_pickle=False).astype(dtype)  # [N, 3390]
    if feat.ndim != 2:
        raise ValueError(f"expect 2D npy, got {feat.shape} from {npy_path}")
    if feat.shape[1] != num_sensors * win_samples:
        raise ValueError(
            f"dim mismatch in {npy_path}: {feat.shape[1]} != {num_sensors}*{win_samples}"
        )

    step = int(round(win_samples * (1.0 - overlap)))  # 30*(1-0.5)=15
    if step <= 0 or step > win_samples:
        raise ValueError(f"bad overlap={overlap} for win_samples={win_samples}")
    ov = win_samples - step  # overlap samples

    # [N, 3390] -> [N, num_sensors, win] -> [N, win, num_sensors]
    x = feat.reshape(-1, num_sensors, win_samples).transpose(0, 2, 1)  # [N, 30, 113]

    out = [x[0]]  # first window full 30 frames
    if x.shape[0] > 1:
        out.append(x[1:, ov:, :].reshape(-1, num_sensors))  # append only new frames
    raw = np.concatenate(out, axis=0)  # [T_frames, 113]
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
    Message raw Message（30Hz）Message mean/var。
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

            m = (x != 0.0)  # [T,C]

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
    folds=(0, 1, 2, 3),
    annotations_dirname="annotations",
    stats_dirname="loso_norm_stats_json",
    fps: int = 30,
    num_sensors: int = 113,
    npy_rel_dir: str = os.path.join("processed", "inertial_features", "30_samples_50_overlap"),
    win_samples: int = 30,
    overlap: float = 0.5,
    ignore_zeros_in_stats: bool = False,
):
    """
    Message 4 Message fold Message stats Message：
      raw/loso_norm_stats_json/loso_sbj_k_stats.json

    Message stats Message fold Message json Message -> Message。
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
    """
    Message fold Message stats json，Message mean/std，Message。
    """
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


# ============================================================
# 5) Optional: a tiny CLI to precompute stats once
# ============================================================
if __name__ == "__main__":

    dataset_dir = "/home/lipei/TAL_data/opportunity/"
    ensure_all_loso_stats_json(
        dataset_dir=dataset_dir,
        folds=(0, 1, 2, 3),
        stats_dirname="loso_norm_stats_json",
        fps=30,
        num_sensors=113,
        npy_rel_dir=os.path.join("processed", "inertial_features", "30_samples_50_overlap"),
        win_samples=30,
        overlap=0.5,
        ignore_zeros_in_stats=False,
    )
    print("Done. Stats saved under:", _stats_dir_in_raw(dataset_dir, "loso_norm_stats_json"))

# if __name__ == "__main__":
#     dataset_dir = "/data/bx_data/TAL_data/hangtime/"
#     ensure_all_loso_stats_json(
#         dataset_dir=dataset_dir,
#         folds=tuple(range(24)),
#         stats_dirname="loso_norm_stats_json",
#         fps=50,
#         num_sensors=3,
#         npy_rel_dir=os.path.join("processed", "inertial_features", "50_samples_50_overlap"),
#         win_samples=50,
#         overlap=0.5,
#         ignore_zeros_in_stats=False,
#     )
#     print("Done. Stats saved under:", _stats_dir_in_raw(dataset_dir, "loso_norm_stats_json"))


# if __name__ == "__main__":
#     dataset_dir = "/home/lipei/TAL_data/rwhar/"
#     ensure_all_loso_stats_json(
#         dataset_dir=dataset_dir,
#         folds=tuple(range(15)),
#         stats_dirname="loso_norm_stats_json",
#         fps=50,
#         num_sensors=21,
#         npy_rel_dir=os.path.join("processed", "inertial_features", "50_samples_50_overlap"),
#         win_samples=50,
#         overlap=0.5,
#         ignore_zeros_in_stats=False,
#     )
#     print("Done. Stats saved under:", _stats_dir_in_raw(dataset_dir, "loso_norm_stats_json"))


# if __name__ == "__main__":
#     dataset_dir = "/data/bx_data/TAL_data/sbhar/"
#     ensure_all_loso_stats_json(
#         dataset_dir=dataset_dir,

#         stats_dirname="loso_norm_stats_json",
#         fps=50,
#         num_sensors=3,
#         npy_rel_dir=os.path.join("processed", "inertial_features", "50_samples_50_overlap"),
#         win_samples=50,
#         overlap=0.5,
#         ignore_zeros_in_stats=False,
#     )
#     print("Done. Stats saved under:", _stats_dir_in_raw(dataset_dir, "loso_norm_stats_json"))


# if __name__ == "__main__":
#     dataset_dir = "/data/bx_data/TAL_data/wear/"
#     ensure_all_loso_stats_json(
#         dataset_dir=dataset_dir,
#         folds=tuple(range(18)),
#         stats_dirname="loso_norm_stats_json",
#         fps=50,
#         num_sensors=12,
#         npy_rel_dir=os.path.join("processed", "inertial_features", "50_samples_50_overlap"),
#         win_samples=50,
#         overlap=0.5,
#         ignore_zeros_in_stats=False,
#     )
#     print("Done. Stats saved under:", _stats_dir_in_raw(dataset_dir, "loso_norm_stats_json"))



# if __name__ == "__main__":
#     dataset_dir = "/data/bx_data/TAL_data/wetlab/"
#     ensure_all_loso_stats_json(
#         dataset_dir=dataset_dir,
#         folds=tuple(range(22)),
#         stats_dirname="loso_norm_stats_json",
#         fps=50,
#         num_sensors=3,
#         npy_rel_dir=os.path.join("processed", "inertial_features", "50_samples_50_overlap"),
#         win_samples=50,
#         overlap=0.5,
#         ignore_zeros_in_stats=False,
#     )
#     print("Done. Stats saved under:", _stats_dir_in_raw(dataset_dir, "loso_norm_stats_json"))
