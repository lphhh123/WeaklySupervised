# dataset_rwhar_ws.py
from OtherData.utils import *
from OtherData.utils import _load_loso_json, _parse_fold_id_from_loso_name, _clip_multihot_label, _subjects_by_subset

class WeaklyRWHARDataset(Dataset):
    """
    弱监督dataset：返回 raw clip + clip级 multi-hot label
    Return:
      x: FloatTensor [C, T]
      y: FloatTensor [num_classes]
      meta(optional)
    """
    def __init__(
        self,
        dataset_dir: str,
        loso_json: str,
        mode: str = "train",  # "train" | "test_window" | "test_full"

        fps: int = 50,
        num_sensors: int = 21,

        npy_rel_dir: str = os.path.join("processed", "inertial_features", "50_samples_50_overlap"),
        win_samples: int = 50,
        overlap: float = 0.5,

        clip_sec: float = 1000.0,
        clip_overlap: float = 0.5,

        num_classes: int = 8,
        min_ov_frames: int = 1,

        neg_keep_ratio: float = 0.2,
        seed: int = 2024,

        normalize: bool = True,
        stats_dirname: str = "loso_norm_stats_json",
        eps: float = 1e-6,

        cache_raw: bool = False,   # ★RWHAR长序列建议 False
        return_meta: bool = False,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.fps = int(fps)
        self.num_sensors = int(num_sensors)
        self.num_classes = int(num_classes)
        self.min_ov_frames = int(min_ov_frames)
        self.neg_keep_ratio = float(neg_keep_ratio)
        self.rng = np.random.RandomState(seed)

        self.npy_rel_dir = npy_rel_dir
        self.win_samples = int(win_samples)
        self.overlap = float(overlap)

        self.return_meta = return_meta
        self.cache_raw = bool(cache_raw)

        ann_path = os.path.join(dataset_dir, "annotations", loso_json) if not os.path.isabs(loso_json) else loso_json
        loso_db, label_dict = _load_loso_json(ann_path)
        self.loso_db = loso_db
        self.label_dict = label_dict

        fold_id = _parse_fold_id_from_loso_name(loso_json)
        if fold_id is None:
            raise RuntimeError(f"loso_json name must be like loso_sbj_0.json, got {loso_json}")
        self.fold_id = fold_id

        # subjects
        if mode == "train":
            self.subjects = _subjects_by_subset(loso_db, "Training")
        else:
            self.subjects = _subjects_by_subset(loso_db, "Validation")

        # mean/std
        self.normalize = bool(normalize)
        if self.normalize:
            self.mean, self.std = load_mean_std_from_stats_json(dataset_dir, stats_dirname, fold_id, eps=eps)
            assert self.mean.shape[0] == self.num_sensors
            assert self.std.shape[0] == self.num_sensors
        else:
            self.mean, self.std = None, None

        # load raw (restored from npy)
        self._raw = {}
        self._raw_path = {}
        npy_dir = os.path.join(dataset_dir, npy_rel_dir)

        for sbj in self.subjects:
            npy_path = os.path.join(npy_dir, f"{sbj}.npy")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(npy_path)
            if self.cache_raw:
                self._raw[sbj] = npy_windows_to_raw_frames(
                    npy_path=npy_path,
                    num_sensors=self.num_sensors,
                    win_samples=self.win_samples,
                    overlap=self.overlap,
                    dtype=np.float32,
                )
            else:
                self._raw_path[sbj] = npy_path

        # build sample index
        self.clip_len = int(round(clip_sec * self.fps))
        self.clip_stride = max(1, int(round(self.clip_len * (1.0 - clip_overlap))))

        self.index = []  # list of (sbj, clip_s, clip_e)
        for sbj in self.subjects:
            raw = self._get_raw(sbj)
            T = raw.shape[0]

            if mode == "test_full":
                self.index.append((sbj, 0, T))
                continue

            if T < self.clip_len:
                continue
            max_s = T - self.clip_len
            for cs in range(0, max_s + 1, self.clip_stride):
                ce = cs + self.clip_len
                y = _clip_multihot_label(self.loso_db[sbj]["annos"], cs, ce, self.num_classes, self.min_ov_frames)
                if mode == "train" and y.sum() == 0:
                    if self.rng.rand() > self.neg_keep_ratio:
                        continue
                self.index.append((sbj, cs, ce))

        if len(self.index) == 0:
            raise RuntimeError(f"No samples built for mode={mode}. Check params.")

    def _get_raw(self, sbj: str) -> np.ndarray:
        if self.cache_raw:
            return self._raw[sbj]
        if sbj not in self._raw:
            npy_path = self._raw_path[sbj]
            self._raw[sbj] = npy_windows_to_raw_frames(
                npy_path=npy_path,
                num_sensors=self.num_sensors,
                win_samples=self.win_samples,
                overlap=self.overlap,
                dtype=np.float32,
            )
        return self._raw[sbj]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        sbj, cs, ce = self.index[idx]
        raw = self._get_raw(sbj)            # [T,C]
        x = raw[cs:ce].T.copy()             # [C,clip_len]

        if self.normalize:
            x = (x - self.mean[:, None]) / (self.std[:, None] + 1e-6)

        y = _clip_multihot_label(self.loso_db[sbj]["annos"], cs, ce, self.num_classes, self.min_ov_frames)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.return_meta:
            return x, y, {"sbj": sbj, "start": cs, "end": ce, "fold_id": self.fold_id}
        return x, y
