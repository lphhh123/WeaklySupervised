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


        self.use_airpods = getattr(cfg, 'USE_AIRPODS', False)


        self.stats_path = os.path.join(cfg.DATA_PATH, 'global_stats.json')

        if self.mode == 'train':
            self.h5_path = os.path.join(cfg.DATA_PATH, 'train_data.h5')
            self.label_path = os.path.join(cfg.DATA_PATH, 'train_label.json')
        else:
            self.csv_path = os.path.join(cfg.DATA_PATH, 'test.csv')
            self.anno_path = cfg.GT_PATH
            self.test_root = cfg.TEST_DATA_ROOT  # .../WWADL/imu



            dataset_root = os.path.dirname(cfg.TEST_DATA_ROOT.rstrip('/'))
            self.airpods_root = os.path.join(dataset_root, 'AirPodsPro')
            if not os.path.exists(self.airpods_root):
                self.airpods_root = os.path.join(dataset_root, 'airpods')


        self.load_global_stats()


        if self.mode == 'train':
            self._init_train()
        else:
            self._init_test()

    def load_global_stats(self):

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
        print(f"Loading Train Labels from {self.label_path}")
        with open(self.label_path, 'r') as f:
            full_labels = json.load(f)


        if self.modal in full_labels:
            self.raw_labels = full_labels[self.modal]
        else:
            self.raw_labels = full_labels


        self.sample_keys = sorted(list(self.raw_labels.keys()), key=lambda x: int(x))
        print(f"=> Train set has {len(self.sample_keys)} sequences")

        print(f"Loading Train H5 from {self.h5_path}")
        with h5py.File(self.h5_path, 'r') as f:

            key = self.modal if self.modal in f else list(f.keys())[0]
            imu_data = f[key][:]
            if len(imu_data.shape) == 4:
                N, T, D, C = imu_data.shape
                imu_data = imu_data.reshape(N, T, D * C)
            self.train_imu_cache = imu_data


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
        for file_name in self.file_names:

            if not file_name.endswith('.h5'):
                h5_name = file_name + '.h5'
            else:
                h5_name = file_name


            file_path = os.path.join(self.test_root, h5_name)
            if not os.path.exists(file_path):
                print(f"⚠️ 跳过文件（未找到）: {file_path}")
                continue


            with h5py.File(file_path, 'r') as f:
                if 'data' in f:
                    raw_imu = f['data'][:]

                    if raw_imu.shape[0] == 5 and raw_imu.shape[1] != 5:
                        raw_imu = np.transpose(raw_imu, (1, 0, 2))
                elif 'amp' in f:
                    raw_imu = f['amp'][:]
                else:
                    raw_imu = f[list(f.keys())[0]][:]

            t_origin = raw_imu.shape[0]


            raw_air = None
            if self.use_airpods:
                air_path = os.path.join(self.airpods_root, h5_name)
                if os.path.exists(air_path):
                    with h5py.File(air_path, 'r') as f:
                        if 'data' in f:

                            raw_air = f['data'][:]
                else:
                    print(f"⚠️ 未找到对应的 AirPods 文件，将补零: {air_path}")
                    raw_air = np.zeros((t_origin, 9))


            def window_generator(t_total, imu_full, air_full):

                if t_total <= clip_length:
                    offsets = [0]
                else:
                    offsets = list(range(0, t_total - clip_length + 1, stride))

                    if offsets[-1] != t_total - clip_length:
                        offsets.append(t_total - clip_length)

                for start_f in offsets:
                    end_f = start_f + clip_length


                    imu_chunk = imu_full[start_f:end_f]

                    imu_feat = self._preprocess_imu(imu_chunk)


                    if self.use_airpods and air_full is not None:



                        air_chunk = air_full[start_f:end_f] if air_full.shape[0] == t_total else air_full


                        if air_chunk.shape[0] != imu_chunk.shape[0]:
                            x_old = np.linspace(0, 1, air_chunk.shape[0])
                            x_new = np.linspace(0, 1, imu_chunk.shape[0])

                            f_interp = interp1d(x_old, air_chunk, axis=0, kind='linear', fill_value="extrapolate")
                            air_chunk = f_interp(x_new)


                        air_feat = self._preprocess_airpods(air_chunk)


                        sample = np.concatenate([imu_feat, air_feat], axis=-1)
                    else:
                        sample = imu_feat


                    yield torch.from_numpy(sample).float().unsqueeze(0), start_f


            yield h5_name, window_generator(t_origin, raw_imu, raw_air), t_origin


    def _preprocess_imu(self, sample_np):
        # sample_np: [T, 5, 6] (Test) or [2048, 5, 6] (Train)


        if len(sample_np.shape) == 3 and sample_np.shape[1] == 5:
            # (T, 5, 6) -> (5, 6, T) -> (30, T)

            sample = np.transpose(sample_np, (1, 2, 0))  # (5, 6, T)
            sample = sample.reshape(30, -1)  # (30, T)
            sample = sample.transpose(1, 0)  # (T, 30)
        else:

            sample = sample_np



        sample = (sample - self.imu_mean) / (self.imu_std + 1e-6)
        return sample

    def _preprocess_airpods(self, sample_np):
        # sample_np: [T, 9]

        sample = sample_np[:, 3:9]  # [T, 6]


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


        imu_path = os.path.join(self.test_root, h5_name)
        with h5py.File(imu_path, 'r') as f:
            if 'data' in f:
                raw_data = f['data'][:]  # [T, 5, 6]
                if raw_data.shape[0] == 5:
                    raw_data = np.transpose(raw_data, (1, 0, 2))
            else:
                raw_data = f[list(f.keys())[0]][:]

        imu_feat = self._preprocess_imu(raw_data)  # [T, 30]
        T_origin = imu_feat.shape[0]

        sample = imu_feat


        if self.use_airpods:
            air_path = os.path.join(self.airpods_root, h5_name)
            if os.path.exists(air_path):
                with h5py.File(air_path, 'r') as f:
                    if 'data' in f:
                        air_raw = f['data'][:]  # [T_air, 9]


                        if air_raw.shape[0] != T_origin:
                            x_old = np.linspace(0, 1, air_raw.shape[0])
                            x_new = np.linspace(0, 1, T_origin)
                            f_interp = interp1d(x_old, air_raw, axis=0, kind='linear', fill_value="extrapolate")
                            air_raw = f_interp(x_new)
                        # -----------------------------------

                        air_feat = self._preprocess_airpods(air_raw)  # [T, 6]
                        sample = np.concatenate([sample, air_feat], axis=-1)
            else:

                air_zeros = np.zeros((T_origin, 6))
                sample = np.concatenate([sample, air_zeros], axis=-1)


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