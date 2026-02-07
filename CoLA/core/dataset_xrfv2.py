import os
import json
import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from scipy.interpolate import interp1d
from core.config_xrfv2 import cfg


class XRFV2Dataset(data.Dataset):
    def __init__(self, mode, modal, num_segments, class_dict, seed=-1, supervision='weak'):
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.class_name_to_idx = class_dict
        self.num_classes = len(class_dict)
        self.supervision = supervision

        # 从 config 读取是否使用 AirPods
        self.use_airpods = getattr(cfg, 'USE_AIRPODS', False)

        # --- 路径配置 ---
        self.stats_path = os.path.join(cfg.DATA_PATH, 'global_stats.json')

        if self.mode == 'train':
            self.h5_path = os.path.join(cfg.DATA_PATH, 'train_data.h5')
            self.label_path = os.path.join(cfg.DATA_PATH, 'train_label.json')
        else:
            self.csv_path = os.path.join(cfg.DATA_PATH, 'test.csv')
            self.anno_path = cfg.GT_PATH
            self.test_root = cfg.TEST_DATA_ROOT  # .../WWADL/imu

            # 推断 AirPods 路径
            # 学姐代码逻辑：dataset_root/AirPodsPro 或 dataset_root/airpods
            dataset_root = os.path.dirname(cfg.TEST_DATA_ROOT.rstrip('/'))
            self.airpods_root = os.path.join(dataset_root, 'AirPodsPro')
            if not os.path.exists(self.airpods_root):
                self.airpods_root = os.path.join(dataset_root, 'airpods')

        # 加载统计数据
        self.load_global_stats()

        # 初始化
        if self.mode == 'train':
            self._init_train()
        else:
            self._init_test()

    def load_global_stats(self):
        # 复用学姐逻辑，分别加载 imu 和 airpods 的 mean/std
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                stats = json.load(f)

            # IMU Stats
            if 'imu' in stats:
                self.imu_mean = np.array(stats['imu']['global_mean'], dtype=np.float32)
                self.imu_std = np.array(stats['imu']['global_std'], dtype=np.float32)
            else:
                self.imu_mean, self.imu_std = 0, 1

            # AirPods Stats
            if self.use_airpods:
                if 'airpods' in stats:
                    self.air_mean = np.array(stats['airpods']['global_mean'], dtype=np.float32)
                    self.air_std = np.array(stats['airpods']['global_std'], dtype=np.float32)
                else:
                    self.air_mean, self.air_std = 0, 1
        else:
            self.imu_mean, self.imu_std = 0, 1
            self.air_mean, self.air_std = 0, 1

    def _init_train(self):
        """[回滚版] 初始化训练数据：直接加载 H5 和 全部弱标签"""
        print(f"Loading Train Labels from {self.label_path}")
        with open(self.label_path, 'r') as f:
            full_labels = json.load(f)

        # 剥离模态外层
        if self.modal in full_labels:
            self.raw_labels = full_labels[self.modal]
        else:
            self.raw_labels = full_labels

        # 恢复逻辑：直接取 JSON 里的所有 key，不看 train.csv
        self.sample_keys = sorted(list(self.raw_labels.keys()), key=lambda x: int(x))
        print(f"=> Train set has {len(self.sample_keys)} sequences")

        print(f"Loading Train H5 from {self.h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            # 读取 IMU
            key = self.modal if self.modal in f else list(f.keys())[0]
            imu_data = f[key][:]
            if len(imu_data.shape) == 4:
                N, T, D, C = imu_data.shape
                imu_data = imu_data.reshape(N, T, D * C)
            self.train_imu_cache = imu_data

            # 读取 AirPods
            if self.use_airpods and 'airpods' in f:
                self.train_air_cache = f['airpods'][:]
            else:
                self.train_air_cache = None

    def _init_test(self):
        print(f"Loading Test List from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if 'file_name' in df.columns:
            self.file_names = df['file_name'].tolist()
        else:
            self.file_names = df.iloc[:, 0].astype(str).tolist()

        if os.path.exists(self.anno_path):
            with open(self.anno_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
        print(f"=> Test set has {len(self.file_names)} files")

    def dataset_windowed(self, clip_length=2048, stride=512):
        """
        完全对齐学姐的 test_window 数据生成逻辑。
        支持 36 维 (IMU 30 + AirPods 6) 数据拼接与时序对齐。
        """
        for file_name in self.file_names:
            # 1. 统一文件名后缀
            if not file_name.endswith('.h5'):
                h5_name = file_name + '.h5'
            else:
                h5_name = file_name

            # 2. 路径对齐
            file_path = os.path.join(self.test_root, h5_name)
            if not os.path.exists(file_path):
                print(f"⚠️ 跳过文件（未找到）: {file_path}")
                continue

            # 3. 读取 IMU 原始数据
            with h5py.File(file_path, 'r') as f:
                if 'data' in f:
                    raw_imu = f['data'][:]  # 预期形状 [T, 5, 6]
                    # 确保维度顺序为 (T, Device, Channel)
                    if raw_imu.shape[0] == 5 and raw_imu.shape[1] != 5:
                        raw_imu = np.transpose(raw_imu, (1, 0, 2))
                elif 'amp' in f:
                    raw_imu = f['amp'][:]
                else:
                    raw_imu = f[list(f.keys())[0]][:]

            t_origin = raw_imu.shape[0]

            # 4. 读取 AirPods 原始数据 (如果开启)
            raw_air = None
            if self.use_airpods:
                air_path = os.path.join(self.airpods_root, h5_name)
                if os.path.exists(air_path):
                    with h5py.File(air_path, 'r') as f:
                        if 'data' in f:
                            # 采样点可能不同，先存原始数据，在 window 循环里做插值
                            raw_air = f['data'][:]  # 预期形状 [T_air, 9]
                else:
                    print(f"⚠️ 未找到对应的 AirPods 文件，将补零: {air_path}")
                    raw_air = np.zeros((t_origin, 9))  # 占位

            # 5. 定义内部生成器：产生该视频的所有滑动窗口
            def window_generator(t_total, imu_full, air_full):
                # 计算滑动起始点
                if t_total <= clip_length:
                    offsets = [0]
                else:
                    offsets = list(range(0, t_total - clip_length + 1, stride))
                    # 确保覆盖视频末尾
                    if offsets[-1] != t_total - clip_length:
                        offsets.append(t_total - clip_length)

                for start_f in offsets:
                    end_f = start_f + clip_length

                    # --- 处理 IMU 窗口 ---
                    imu_chunk = imu_full[start_f:end_f]
                    # 调用类内置预处理 (转置为 [T, 30] + 归一化)
                    imu_feat = self._preprocess_imu(imu_chunk)

                    # --- 处理 AirPods 窗口并拼接 ---
                    if self.use_airpods and air_full is not None:
                        # 截取 AirPods 对应的切片
                        # 注意：如果 AirPods 采样率不同，这里的索引可能需要缩放
                        # 但根据 XRFV2 惯例，h5 内部已基本对齐，若长度微差则插值
                        air_chunk = air_full[start_f:end_f] if air_full.shape[0] == t_total else air_full

                        # 如果切片长度与 IMU 窗口不一致，进行插值对齐
                        if air_chunk.shape[0] != imu_chunk.shape[0]:
                            x_old = np.linspace(0, 1, air_chunk.shape[0])
                            x_new = np.linspace(0, 1, imu_chunk.shape[0])
                            # 对 9 维全部插值
                            f_interp = interp1d(x_old, air_chunk, axis=0, kind='linear', fill_value="extrapolate")
                            air_chunk = f_interp(x_new)

                        # 调用类内置预处理 (取 3:9 维 + 归一化)
                        air_feat = self._preprocess_airpods(air_chunk)

                        # 拼接成 36 维: [T, 30] + [T, 6] -> [T, 36]
                        sample = np.concatenate([imu_feat, air_feat], axis=-1)
                    else:
                        sample = imu_feat

                    # 返回模型需要的形状: [1, T, 36] Tensor 和全局起始帧
                    yield torch.from_numpy(sample).float().unsqueeze(0), start_f

            # 返回给外部: 视频名, 窗口迭代器, 视频原始总长度
            yield h5_name, window_generator(t_origin, raw_imu, raw_air), t_origin

    # --- 核心预处理函数 (参考学姐代码) ---
    def _preprocess_imu(self, sample_np):
        # sample_np: [T, 5, 6] (Test) or [2048, 5, 6] (Train)
        # 转置 -> [5, 6, T] -> [30, T]
        # 注意: 学姐代码里 train 是 (2048, 5, 6), test 读取后也是这个形状
        if len(sample_np.shape) == 3 and sample_np.shape[1] == 5:
            # (T, 5, 6) -> (5, 6, T) -> (30, T)
            # 但 CoLA 需要 (T, C)，所以最后要转回来
            sample = np.transpose(sample_np, (1, 2, 0))  # (5, 6, T)
            sample = sample.reshape(30, -1)  # (30, T)
            sample = sample.transpose(1, 0)  # (T, 30)
        else:
            # 已经是 (T, 30) ?
            sample = sample_np

        # 归一化
        # global_mean 是 (30,)，需要广播
        sample = (sample - self.imu_mean) / (self.imu_std + 1e-6)
        return sample

    def _preprocess_airpods(self, sample_np):
        # sample_np: [T, 9]
        # 取 [:, 3:9] -> Acc+Gyro
        sample = sample_np[:, 3:9]  # [T, 6]

        # 归一化
        sample = (sample - self.air_mean) / (self.air_std + 1e-6)
        return sample

    def __len__(self):
        return len(self.sample_keys) if self.mode == 'train' else len(self.file_names)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self._get_train_item(index)
        else:
            return self._get_test_item(index)

    def _get_train_item(self, index):
        # 1. IMU
        imu_sample = self.train_imu_cache[index]  # [2048, 5, 6]
        imu_feat = self._preprocess_imu(imu_sample)  # [2048, 30]

        # 2. AirPods
        sample = imu_feat
        if self.use_airpods and self.train_air_cache is not None:
            air_sample = self.train_air_cache[index]  # [2048, 9]
            air_feat = self._preprocess_airpods(air_sample)  # [2048, 6]
            sample = np.concatenate([imu_feat, air_feat], axis=-1)  # [2048, 36]

        # 采样 (CoLA)
        T, C = sample.shape
        if self.num_segments > 0 and self.num_segments != T:
            indices = np.linspace(0, T - 1, self.num_segments).astype(int)
            sample = sample[indices]

        sample = torch.from_numpy(sample).float()

        # Label
        key = self.sample_keys[index]
        actions = self.raw_labels[key]
        video_level_label = np.zeros(self.num_classes, dtype=np.float32)
        for item in actions:
            if len(item) >= 3:
                cid = int(item[2])
                if 0 <= cid < self.num_classes:
                    video_level_label[cid] = 1.0

        return sample, torch.from_numpy(video_level_label), torch.tensor(0), key, T

    def _get_test_item(self, index):
        file_name = self.file_names[index]
        if not file_name.endswith('.h5'):
            h5_name = file_name + '.h5'
        else:
            h5_name = file_name
            file_name = file_name.replace('.h5', '')

        # 1. 读取 IMU
        imu_path = os.path.join(self.test_root, h5_name)
        with h5py.File(imu_path, 'r') as f:
            if 'data' in f:
                raw_data = f['data'][:]  # [T, 5, 6]
                if raw_data.shape[0] == 5:  # 兼容 (5, T, 6)
                    raw_data = np.transpose(raw_data, (1, 0, 2))
            else:
                raw_data = f[list(f.keys())[0]][:]

        imu_feat = self._preprocess_imu(raw_data)  # [T, 30]
        T_origin = imu_feat.shape[0]

        sample = imu_feat

        # 2. 读取 AirPods (如果开启)
        if self.use_airpods:
            air_path = os.path.join(self.airpods_root, h5_name)
            if os.path.exists(air_path):
                with h5py.File(air_path, 'r') as f:
                    if 'data' in f:
                        air_raw = f['data'][:]  # [T_air, 9]

                        # --- 插值对齐逻辑 (修复 3400 vs 1700 问题) ---
                        if air_raw.shape[0] != T_origin:
                            x_old = np.linspace(0, 1, air_raw.shape[0])
                            x_new = np.linspace(0, 1, T_origin)
                            f_interp = interp1d(x_old, air_raw, axis=0, kind='linear', fill_value="extrapolate")
                            air_raw = f_interp(x_new)
                        # -----------------------------------

                        air_feat = self._preprocess_airpods(air_raw)  # [T, 6]
                        sample = np.concatenate([sample, air_feat], axis=-1)
            else:
                # 补零
                air_zeros = np.zeros((T_origin, 6))
                sample = np.concatenate([sample, air_zeros], axis=-1)

        # 采样 (CoLA 测试也需要固定长度)
        if self.num_segments > 0:
            indices = np.linspace(0, T_origin - 1, self.num_segments).astype(int)
            sample = sample[indices]

        sample = torch.from_numpy(sample).float()

        # Label
        video_level_label = np.zeros(self.num_classes, dtype=np.float32)
        json_key = h5_name
        if 'database' in self.annotations and json_key in self.annotations['database']:
            entry = self.annotations['database'][json_key]
            for ann in entry['annotations']:
                label_name = ann['label']
                if label_name in self.class_name_to_idx:
                    cid = self.class_name_to_idx[label_name]
                    video_level_label[cid] = 1.0

        return sample, torch.from_numpy(video_level_label), torch.tensor(0), h5_name, T_origin