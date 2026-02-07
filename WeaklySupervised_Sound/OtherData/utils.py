import os, json, time, math, random

import torch.nn as nn
from typing import Optional
from typing import Union
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import re


# ============================

# ==============================
def _load_loso_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    db = obj["database"]

    parsed = {}
    for sbj, info in db.items():
        subset = info.get("subset", None)
        fps = int(info.get("fps", 50))
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


def _subjects_by_split(loso_db: dict, split: str):
    if split not in ["train", "test"]:
        raise ValueError("split must be 'train' or 'test'")
    target_subset = "Training" if split == "train" else "Validation"
    subs = [k for k, v in loso_db.items() if v.get("subset") == target_subset]
    subs.sort()
    return subs


def _majority_label_in_window(win_s: int, win_e: int, annos, win_len: int, min_frac: float = 0.0):
    """
    Message [win_s, win_e) Message label Message，Message label。
    - annos: [(s,e,lid), ...] Message s Message，Message [s,e)
    - min_frac: Message < min_frac，Message None（Message）
      Message 0.0：Message label
    """
    if win_e <= win_s:
        return None

    overlap_per_label = {}  # lid -> overlap_frames

    for s, e, lid in annos:
        if s >= win_e:
            break
        if e <= win_s:
            continue
        ov = min(win_e, e) - max(win_s, s)
        if ov > 0:
            overlap_per_label[lid] = overlap_per_label.get(lid, 0) + int(ov)

    if not overlap_per_label:
        return None


    best_lid, best_ov = max(overlap_per_label.items(), key=lambda kv: (kv[1], -kv[0]))
    best_frac = float(best_ov) / float(max(win_len, 1))

    if best_frac < float(min_frac):
        return None
    return best_lid


def _parse_fold_id_from_loso_name(loso_json: str):
    base = os.path.basename(loso_json)
    m = re.search(r"loso_sbj_(\d+)", base)
    if m:
        return int(m.group(1))
    return None


def npy_windows_to_raw_frames(
    npy_path: str,
    num_sensors: int = 21,
    win_samples: int = 50,     # 50_samples
    overlap: float = 0.5,      # 50_overlap
    dtype=np.float32,
):
    """
    npy: [N, num_sensors*win_samples]
    return raw: [T_frames, num_sensors]  (e.g., [215825, 21])
    """
    feat = np.load(npy_path, allow_pickle=False).astype(dtype)
    if feat.ndim != 2:
        raise ValueError(f"expect 2D npy, got {feat.shape} from {npy_path}")
    if feat.shape[1] != num_sensors * win_samples:
        raise ValueError(f"dim mismatch: {feat.shape[1]} != {num_sensors}*{win_samples}")

    step = int(round(win_samples * (1.0 - overlap)))  # 25
    if step <= 0 or step > win_samples:
        raise ValueError(f"bad overlap={overlap} for win_samples={win_samples}")
    ov = win_samples - step  # 25

    x = feat.reshape(-1, num_sensors, win_samples).transpose(0, 2, 1)  # [N,50,21]

    out = [x[0]]
    if x.shape[0] > 1:
        out.append(x[1:, ov:, :].reshape(-1, num_sensors))
    raw = np.concatenate(out, axis=0)  # [T,21]
    return raw


# ----------------------------
# Load mean/var JSON
# ----------------------------
def _stats_json_path(dataset_dir: str, fold_id: int, stats_dirname: str):
    return os.path.join(dataset_dir, "raw", stats_dirname, f"loso_sbj_{fold_id}_stats.json")

def load_mean_std_from_meanvar_json(
    dataset_dir: str,
    fold_id: int,
    stats_dirname: str,
    eps: float,
    expect_meta: dict,
):
    """
    Message JSON Message（mean + var）
    Message meta Message，Message fold Message。
    """
    path = _stats_json_path(dataset_dir, fold_id, stats_dirname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stats json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)


    for k, v in expect_meta.items():
        if k not in obj:
            raise RuntimeError(f"Stats json missing key '{k}': {path}")
        if obj[k] != v:
            raise RuntimeError(f"Stats meta mismatch [{k}]: file={obj[k]} vs expect={v} | {path}")

    if "mean" not in obj or "var" not in obj:
        raise RuntimeError(f"Stats json must contain 'mean' and 'var': {path}")

    mean = np.asarray(obj["mean"], dtype=np.float32)
    var = np.asarray(obj["var"], dtype=np.float32)
    if mean.ndim != 1 or var.ndim != 1:
        raise RuntimeError(f"mean/var must be 1D arrays in {path}")
    if mean.shape[0] != var.shape[0]:
        raise RuntimeError(f"mean/var length mismatch in {path}: {mean.shape} vs {var.shape}")

    std = np.sqrt(np.maximum(var, 0.0) + float(eps)).astype(np.float32)
    return mean, std
# ===============================

def set_seed(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _uniform_pick_indices(n: int, k: int):
    """Message n Message k Message index（Message）"""
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    return np.linspace(0, n - 1, k).astype(int).tolist()

def load_mean_std_from_stats_json(dataset_dir: str, stats_dirname: str, fold_id: int, eps=1e-6):
    stats_path = os.path.join(dataset_dir, "raw", stats_dirname, f"loso_sbj_{fold_id}_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"stats json not found: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    mean = np.asarray(js["mean"], dtype=np.float32)
    var  = np.asarray(js["var"], dtype=np.float32)
    std  = np.sqrt(np.maximum(var, 0.0) + eps).astype(np.float32)
    return mean, std

def _meta_get(meta, key):
    v = meta[key]
    if isinstance(v, (list, tuple)):
        v = v[0]
    if torch.is_tensor(v):
        v = v.item()
    return v


def build_gt_for_anet(loso_json_path, out_gt_path):
    """
    Message loso json Message subset=Validation Message test，Message ANETdetection Message
    """
    with open(loso_json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    db = js["database"]
    for vid, info in db.items():
        if info.get("subset") == "Validation":
            info["subset"] = "test"
    os.makedirs(os.path.dirname(out_gt_path), exist_ok=True)
    with open(out_gt_path, "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)


def featbox_to_time_seconds(
    clip_start_frame: int,
    clip_end_frame: int,
    start_idx: int,
    end_idx: int,
    bin_frames: int,
    fps: int,
):
    """
    proposal Message([start_idx,end_idx) in T_global) -> Message
    """
    cs = int(clip_start_frame)
    ce = int(clip_end_frame)

    raw_s = cs + int(start_idx) * int(bin_frames)
    raw_e = cs + int(end_idx) * int(bin_frames)

    raw_s = max(cs, min(raw_s, ce - 1))
    raw_e = max(raw_s + 1, min(raw_e, ce))

    return raw_s / float(fps), raw_e / float(fps)


def generate_proposal_boxes(
    T_global: int,
    num_proposals: int,
    fps: int = 30,

    clip_sec: float = 30.0,
    raw_frames: Optional[int] = None,

    base_physical_sec: float = 7.0,
    step_sec: float = 2.0,
    min_sec: float = 5.0,
    max_sec: float = 15.0,

    seed: int = 2024,


    sec_resolution: float = 1.0,
    fixed_keep_ratio: float = 0.7,
    fixed_per_scale_min: int = 2,
):
    """
    Message proposals: Tensor[P,2]，Message [start,end)。
    Message：Message [min_sec, max_sec] Message（sec_resolution Message）
    """

    rng = random.Random(int(seed))

    if raw_frames is None:
        raw_frames = int(round(float(clip_sec) * float(fps)))
    else:
        raw_frames = int(raw_frames)
        clip_sec = float(raw_frames) / float(fps)

    assert T_global >= 1 and raw_frames >= 1
    raw_to_feat = float(T_global) / float(raw_frames)



    lo = max(1e-6, float(min_sec))
    hi = min(float(max_sec), float(clip_sec))
    if hi < lo:
        hi = lo


    step = max(1e-6, float(sec_resolution))
    n_scales = int(math.floor((hi - lo) / step)) + 1
    scales = [lo + i * step for i in range(n_scales)]


    if float(base_physical_sec) >= lo and float(base_physical_sec) <= hi:
        scales.append(float(base_physical_sec))


    scales = sorted(set([round(s, 6) for s in scales]))


    step_feat = max(1, int(round(float(step_sec) * float(fps) * raw_to_feat)))

    pool_by_scale = {}  # scale_sec -> list[[s,e],...]
    for sec in scales:
        raw_len = int(round(sec * fps))
        feat_len = max(1, int(round(raw_len * raw_to_feat)))

        boxes = []
        s = 0
        while s + feat_len <= T_global:
            boxes.append([s, s + feat_len])
            s += step_feat

        if len(boxes) > 0:
            pool_by_scale[sec] = boxes


    valid_scales = list(pool_by_scale.keys())
    if len(valid_scales) == 0:

        return torch.tensor([[0, T_global]], dtype=torch.long)[:num_proposals]


    num_fixed = int(round(num_proposals * float(fixed_keep_ratio)))
    num_fixed = max(0, min(num_fixed, num_proposals))
    num_rand  = num_proposals - num_fixed

    fixed_props = []

    if num_fixed > 0:

        base_quota = min(fixed_per_scale_min, max(1, num_fixed // len(valid_scales)))
        quotas = {s: 0 for s in valid_scales}


        used = 0
        for s in valid_scales:
            q = min(base_quota, len(pool_by_scale[s]))
            quotas[s] = q
            used += q
            if used >= num_fixed:
                break


        remaining = num_fixed - used
        if remaining > 0:

            idx = 0
            scales_cycle = valid_scales[:]
            while remaining > 0:
                s = scales_cycle[idx % len(scales_cycle)]
                if quotas[s] < len(pool_by_scale[s]):
                    quotas[s] += 1
                    remaining -= 1
                idx += 1

                if idx > 10 * (remaining + 1) * len(scales_cycle):
                    break


        for s in valid_scales:
            q = quotas[s]
            if q <= 0:
                continue
            boxes = pool_by_scale[s]
            pick = _uniform_pick_indices(len(boxes), q)
            fixed_props.extend([boxes[i] for i in pick])


        if len(fixed_props) < num_fixed:
            all_fixed = []
            for s in valid_scales:
                all_fixed.extend(pool_by_scale[s])

            all_fixed = list({(a,b) for a,b in all_fixed})
            all_fixed.sort(key=lambda x: (x[0], x[1]))
            need = num_fixed - len(fixed_props)
            pick = _uniform_pick_indices(len(all_fixed), need)
            fixed_props.extend([list(all_fixed[i]) for i in pick])

    props = fixed_props[:num_fixed]


    while len(props) < num_proposals:
        dur_sec = rng.uniform(lo, hi)
        raw_len = int(round(dur_sec * fps))
        feat_len = max(1, int(round(raw_len * raw_to_feat)))

        max_start_sec = max(0.0, clip_sec - float(dur_sec))
        start_sec = 0.0 if max_start_sec <= 0 else rng.uniform(0.0, max_start_sec)

        feat_start = int(round(start_sec * fps * raw_to_feat))
        feat_end = min(feat_start + feat_len, T_global)
        if feat_end <= feat_start:
            feat_end = min(feat_start + 1, T_global)

        props.append([feat_start, feat_end])

    return torch.tensor(props[:num_proposals], dtype=torch.long)



# ============================================================
# 2)  global backbone wrapper
# ============================================================
class GlobalBackboneWrapper(nn.Module):
    """
    Message 3s backbone:
        [bs, C, 90] -> [bs, 512, 30]
        [B, C, T_frames] -> [B, 512, T_global]
    """

    def __init__(self, backbone: nn.Module, win_len=90, seg_stride=45, chunk=256):
        super().__init__()
        self.backbone = backbone
        self.win_len = int(win_len)
        self.seg_stride = int(seg_stride)
        self.chunk = int(chunk)

    @torch.no_grad()
    def forward(self, x_raw: torch.Tensor, return_info: bool = False):
        """
        x_raw: [B, C, T_frames] (trainMessage T_frames=900)
        return:
          - global_feat: [B, 512, T_global]
          - (optional) info: dict(bin_frames=..., T_global=..., raw_frames=...)
        """
        B, C, T = x_raw.shape
        win_len = self.win_len
        seg_stride = self.seg_stride

        if T < win_len:
            pad = win_len - T
            x_raw = torch.cat([x_raw, x_raw[:, :, -1:].repeat(1, 1, pad)], dim=2)
            T = x_raw.shape[2]

        # [B, C, Tg, win_len]
        x = x_raw.unfold(dimension=2, size=win_len, step=seg_stride)
        Tg = x.shape[2]

        # -> [B*Tg, C, win_len]
        x = x.permute(0, 2, 1, 3).contiguous().view(B * Tg, C, win_len)

        outs = []
        for i in range(0, B * Tg, self.chunk):
            out = self.backbone(x[i:i + self.chunk])  # [bs,512,Lout]
            if out.ndim != 3:
                raise RuntimeError(f"backbone output must be [bs,512,Lout], got {out.shape}")
            outs.append(out)
        outs = torch.cat(outs, dim=0)  # [B*Tg,512,Lout]
        Lout = outs.shape[-1]


        if win_len % Lout != 0:
            raise RuntimeError(f"win_len={win_len} not divisible by Lout={Lout}")
        bin_frames = win_len // Lout  # 90//30 = 3 frame/bin

        if seg_stride % bin_frames != 0:
            raise RuntimeError(f"seg_stride={seg_stride} not divisible by bin_frames={bin_frames}")
        stride_bins = seg_stride // bin_frames  # 45//3 = 15 bins appended each step

        # [B,Tg,512,Lout]
        outs = outs.view(B, Tg, 512, Lout)

        # T_global = Lout + (Tg-1)*stride_bins
        T_global = int(Lout + max(0, Tg - 1) * stride_bins)


        global_feat = torch.empty((B, 512, T_global), device=outs.device, dtype=outs.dtype)


        global_feat[:, :, :Lout] = outs[:, 0, :, :]


        for t in range(1, Tg):
            st = Lout + (t - 1) * stride_bins
            ed = st + stride_bins
            global_feat[:, :, st:ed] = outs[:, t, :, -stride_bins:]

        if return_info:
            info = {
                "bin_frames": int(bin_frames),
                "stride_bins": int(stride_bins),
                "T_global": int(T_global),
                "raw_frames": int(T),
                "Tg_windows": int(Tg),
                "Lout": int(Lout),
            }
            return global_feat.contiguous(), info

        return global_feat.contiguous()


class ProposalWrappedDataset(Dataset):
    def __init__(
        self,
        base_ds,
        num_proposals: int,
        backbone: nn.Module,
        win_len: int = 90,
        seg_stride: int = 45,
        fps: int = 30,
        base_physical_sec: float = 7.0,
        step_sec: float = 2.0,
        min_sec: float = 5.0,
        max_sec: float = 15.0,
        seed: int = 2024,
        device: Union[str, torch.device] = "cpu",
        return_meta: bool = False,
    ):
        self.base_ds = base_ds
        self.num_proposals = int(num_proposals)

        self.win_len = int(win_len)
        self.seg_stride = int(seg_stride)
        self.fps = int(fps)

        self.base_physical_sec = float(base_physical_sec)
        self.step_sec = float(step_sec)
        self.min_sec = float(min_sec)
        self.max_sec = float(max_sec)
        self.seed = int(seed)

        self.return_meta = bool(return_meta)

        # -------- 1) probe Lout --------
        backbone = backbone.to(device)
        backbone.eval()
        with torch.no_grad():

            C = getattr(base_ds, "num_sensors", None)
            if C is None:

                x0, y0, meta0 = base_ds[0]
                C = int(x0.shape[0])
            dummy = torch.zeros(1, C, self.win_len, device=device, dtype=torch.float32)
            out = backbone(dummy)  # [1,512,Lout]
            self.Lout = int(out.shape[-1])

        if self.win_len % self.Lout != 0:
            raise RuntimeError(f"win_len={self.win_len} not divisible by Lout={self.Lout}")
        self.bin_frames = self.win_len // self.Lout

        if self.seg_stride % self.bin_frames != 0:
            raise RuntimeError(f"seg_stride={self.seg_stride} not divisible by bin_frames={self.bin_frames}")
        self.stride_bins = self.seg_stride // self.bin_frames

    def __len__(self):
        return len(self.base_ds)

    def _tglobal_from_rawframes(self, raw_frames: int) -> int:

        T = int(max(raw_frames, self.win_len))
        Tg = (T - self.win_len) // self.seg_stride + 1
        T_global = self.Lout + (Tg - 1) * self.stride_bins
        return int(T_global)

    def __getitem__(self, idx):

        x, y, meta = self.base_ds[idx]        # x:[C,T], y:[K]
        raw_frames = int(x.shape[-1])

        T_global = self._tglobal_from_rawframes(raw_frames)

        props = generate_proposal_boxes(
            T_global=T_global,
            num_proposals=self.num_proposals,
            fps=self.fps,
            raw_frames=max(raw_frames, self.win_len),
            base_physical_sec=self.base_physical_sec,
            step_sec=self.step_sec,
            min_sec=self.min_sec,
            max_sec=self.max_sec,
            seed=self.seed + idx,
        )

        if self.return_meta:
            return x, props, y, meta
        return x, props, y

def _subjects_by_subset(loso_db, subset_name: str):
    # subset_name: "Training" / "Validation"
    return [k for k,v in loso_db.items() if v.get("subset","") == subset_name]

def _clip_multihot_label(annos, clip_s, clip_e, num_classes: int, min_ov_frames: int = 1):
    """
    annos: list[(s,e,lid)] in frames
    clip_s, clip_e: frames
    """
    y = np.zeros((num_classes,), dtype=np.float32)
    for s,e,lid in annos:
        ov = min(e, clip_e) - max(s, clip_s)
        if ov >= min_ov_frames:
            if 0 <= lid < num_classes:
                y[lid] = 1.0
    return y


def _to_jsonable(obj):
    """Message config Message numpy Message json Message Python Message。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):  # np.int64 / np.float32 ...
        return obj.item()
    return str(obj)

def dump_config(config: dict, out_dir: str, filename: str = "config.json"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(config), f, indent=2, ensure_ascii=False)
    print(f"[Saved] config -> {out_path}")
    return out_path