# dataset_opportunity.py


from OtherData.utils import *
from OtherData.utils import _load_loso_json, _parse_fold_id_from_loso_name, _majority_label_in_window, \
    _subjects_by_split


# ----------------------------
# Dataset
# ----------------------------
class OpportunityDataset_3s(Dataset):
    """
    NPY-only 预训练 Dataset：
    - 从 processed/.../sbj_k.npy 还原连续 30Hz raw [T,113]
    -  3s window (stride 50%)
    - window center frame 落在 GT 段内 -> 占比大的label确定为此window的label_id
    - 归一化：读取 raw/<stats_dirname>/loso_sbj_{fold}_stats.json （mean+var）
    """
    def __init__(
        self,
        dataset_dir: str,
        loso_json: str,
        split: str = "train",
        subjects=None,

        fps: int = 30,
        num_sensors: int = 113,

        npy_rel_dir: str = os.path.join("processed", "inertial_features", "30_samples_50_overlap"),
        win_samples: int = 30,
        overlap: float = 0.5,

        win_sec: float = 3.0,
        win_overlap: float = 0.5,

        normalize: bool = True,
        stats_dirname: str = "loso_norm_stats_json",
        ignore_zeros_in_stats: bool = False,  # 需要与 stats json 里的字段一致
        eps: float = 1e-6,
        cache_raw: bool = True,
        return_meta: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.return_meta = return_meta
        self.fps = int(fps)
        self.num_sensors = int(num_sensors)

        self.npy_rel_dir = npy_rel_dir
        self.win_samples = int(win_samples)
        self.overlap = float(overlap)

        # 1) load loso annotations
        ann_path = os.path.join(dataset_dir, "annotations", loso_json) if not os.path.isabs(loso_json) else loso_json
        self.loso_db = _load_loso_json(ann_path)

        # 2) choose subjects
        if subjects is None:
            subjects = _subjects_by_split(self.loso_db, split)
        self.subjects = list(subjects)
        self.split = split

        # 3) window settings (frames)
        self.win_len = int(round(win_sec * self.fps))
        self.win_stride = max(1, int(round(self.win_len * (1.0 - win_overlap))))
        # 窗口内占比阈值（默认 0.0 表示只要有 overlap 就分配 label）
        self.min_label_frac = 0.0

        # 4) load raw restored from npy
        self.cache_raw = cache_raw
        self._raw = {}
        self._raw_path = {}
        self._raw_cache = {}

        npy_dir = os.path.join(dataset_dir, npy_rel_dir)
        for sbj in self.subjects:
            npy_path = os.path.join(npy_dir, f"{sbj}.npy")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"NPY not found: {npy_path}")

            if cache_raw:
                self._raw[sbj] = npy_windows_to_raw_frames(
                    npy_path=npy_path,
                    num_sensors=self.num_sensors,
                    win_samples=self.win_samples,
                    overlap=self.overlap,
                    dtype=np.float32,
                )
            else:
                self._raw_path[sbj] = npy_path

        # 5) load mean/std from fold-specific stats json
        self.normalize = bool(normalize)
        self.mean = None
        self.std = None
        self.stats_dirname = stats_dirname
        self.ignore_zeros_in_stats = bool(ignore_zeros_in_stats)
        self.eps = float(eps)

        if self.normalize:
            fold_id = _parse_fold_id_from_loso_name(loso_json)
            if fold_id is None:
                raise RuntimeError(
                    f"Cannot parse fold_id from loso_json name: {loso_json}. "
                    f"请保持命名如 loso_sbj_0.json"
                )
            self.fold_id = fold_id

            expect_meta = {
                "src": "npy",
                "fold_id": int(fold_id),
                "fps": int(self.fps),
                "num_sensors": int(self.num_sensors),
                "npy_rel_dir": str(self.npy_rel_dir).replace("\\", "/"),
                "win_samples": int(self.win_samples),
                "overlap": float(self.overlap),
                "ignore_zeros_in_stats": bool(self.ignore_zeros_in_stats),
            }

            mean, std = load_mean_std_from_meanvar_json(
                dataset_dir=dataset_dir,
                fold_id=fold_id,
                stats_dirname=stats_dirname,
                eps=self.eps,
                expect_meta=expect_meta,
            )

            if mean.shape[0] != self.num_sensors:
                raise RuntimeError(f"mean length={mean.shape[0]} != num_sensors={self.num_sensors}")
            if std.shape[0] != self.num_sensors:
                raise RuntimeError(f"std length={std.shape[0]} != num_sensors={self.num_sensors}")

            self.mean, self.std = mean, std
        else:
            self.fold_id = None

        if self.normalize:
            # 缓存成 torch tensor，避免每个 __getitem__ 重复 from_numpy
            self.mean_t = torch.from_numpy(self.mean).float().unsqueeze(1)  # [C,1]
            self.std_t = torch.from_numpy(self.std).float().unsqueeze(1)  # [C,1]
        else:
            self.mean_t, self.std_t = None, None

        # 6) build index list: (sbj, s, e, lid) —— 去掉 clip，直接整段滑 3s window
        self.index = []
        for sbj in self.subjects:
            annos = self.loso_db[sbj]["annos"]
            T = self._get_raw_len(sbj)
            if T < self.win_len:
                continue

            for s in range(0, T - self.win_len + 1, self.win_stride):
                e = s + self.win_len

                lid = _majority_label_in_window(
                    win_s=s,
                    win_e=e,
                    annos=annos,
                    win_len=self.win_len,
                    min_frac=getattr(self, "min_label_frac", 0.0),
                )
                if lid is None:
                    continue  # 背景/占比不足 直接丢弃
                self.index.append((sbj, s, e, lid))

        if len(self.index) == 0:
            raise RuntimeError("No labeled 3s windows found. Check NPY->raw alignment and annotations.")

    def _get_raw_array(self, sbj: str) -> np.ndarray:
        if self.cache_raw:
            return self._raw[sbj]
        if sbj not in self._raw_cache:
            npy_path = self._raw_path[sbj]
            self._raw_cache[sbj] = npy_windows_to_raw_frames(
                npy_path=npy_path,
                num_sensors=self.num_sensors,
                win_samples=self.win_samples,
                overlap=self.overlap,
                dtype=np.float32,
            )
        return self._raw_cache[sbj]

    def _get_raw_len(self, sbj: str) -> int:
        return int(self._get_raw_array(sbj).shape[0])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        sbj, s, e, lid = self.index[idx]
        raw = self._get_raw_array(sbj)          # [T,C]
        x = raw[s:e, :]                         # [win_len, C]
        x = torch.from_numpy(x.T.copy()).float()  # -> [C, T]

        if self.normalize:
            x = (x - self.mean_t) / self.std_t

        y = torch.tensor(lid, dtype=torch.long) #动作编号

        if self.return_meta:
            return x, y, {"sbj": sbj, "start": s, "end": e, "fold_id": self.fold_id}
        return x, y

