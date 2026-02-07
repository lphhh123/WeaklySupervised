from scipy.interpolate import interp1d
import numpy as np
from models.WSDDN_model import *


class WeaklySupervisedXRFV2DatasetTrain(Dataset):
    def __init__(
        self,
        dataset_dir,
        mapping_path,
        split="train",
        num_proposals=50,
        merge_proposals=False,
        T_global=30,
        use_airpods=True,     # ★ 是否使用 airpods
    ):
        # ★ 调试开关：只打印第一条
        self._debug_print = False

        self.dataset_dir = dataset_dir
        self.split = split
        self.num_proposals = num_proposals
        self.merge_proposals = merge_proposals
        self.modality = "imu"
        self.T_global = T_global
        self.use_airpods = use_airpods

        # 加载标签映射
        self.id_to_action, self.old_to_new, self.old_to_new_action = load_label_mapping(mapping_path)

        # 加载30s IMU数据和锚框标签
        self.data_path = os.path.join(dataset_dir, f"{split}_data.h5")
        self.label_path = os.path.join(dataset_dir, f"{split}_label.json")
        with open(self.label_path, 'r') as f:
            self.anchor_labels = json.load(f)[self.modality]  # {样本ID: 锚框列表}

        # 加载全局归一化参数
        self.global_mean_imu, self.global_std_imu = self._load_stats('imu')
        if self.use_airpods:
            self.global_mean_air, self.global_std_air = self._load_stats('airpods')

    def _load_stats(self, modality='imu'):
        stats_path = os.path.join(self.dataset_dir, "global_stats.json")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        if modality not in stats:
            raise KeyError(f"global_stats.json 中没有 {modality} 的统计量")
        return (
            np.array(stats[modality]["global_mean"]),
            np.array(stats[modality]["global_std"]),
        )

    def _get_video_level_label(self, sample_idx):
        """从锚框标签中提取视频级标签（映射为新ID）"""
        anchor_list = self.anchor_labels[str(sample_idx)]
        old_ids = list({int(anchor[2]) for anchor in anchor_list})  # 去重旧ID
        new_ids = [self.old_to_new[old_id] for old_id in old_ids]
        # 生成one-hot标签
        label = torch.zeros(len(self.old_to_new_action))
        for new_id in new_ids:
            label[new_id] = 1.0
        return label

    def __len__(self):
        with h5py.File(self.data_path, 'r') as f:
            return f[self.modality].shape[0]

    # 封装 imu / airpods 的预处理
    def _preprocess_imu(self, sample_30s_np):
        """
        sample_30s_np: (2048, 5, 6)
        -> [5, 6, 2048] -> [30, 2048]，再用 imu 的 global_mean/_std 归一化
        """
        sample_30s = torch.tensor(sample_30s_np, dtype=torch.float32)  # (2048, 5, 6)
        sample_30s = sample_30s.permute(1, 2, 0).reshape(30, -1)       # [30, 2048]

        mean = torch.tensor(self.global_mean_imu, dtype=torch.float32)[:, None]
        std = torch.tensor(self.global_std_imu, dtype=torch.float32)[:, None] + 1e-6
        sample_30s = (sample_30s - mean) / std                         # [30, 2048]
        return sample_30s

    def _preprocess_airpods(self, sample_air_np):
        """
        sample_air_np: (2048, 9)
        只取加速度(3:6) + 角速度(6:9) 共6维:
        -> [2048, 6] -> [6, 2048]，再用 airpods 的 global_mean/_std 归一化
        """
        sample_air = torch.tensor(sample_air_np, dtype=torch.float32)  # (2048, 9)
        acceleration = sample_air[:, 3:6]  # [2048, 3]
        rotation = sample_air[:, 6:9]      # [2048, 3]
        sample_air = torch.cat((acceleration, rotation), dim=-1)      # [2048, 6]
        sample_air = sample_air.T                                      # [6, 2048]

        mean = torch.tensor(self.global_mean_air, dtype=torch.float32)[:, None]
        std = torch.tensor(self.global_std_air, dtype=torch.float32)[:, None] + 1e-6
        sample_air = (sample_air - mean) / std                        # [6, 2048]
        return sample_air


    def __getitem__(self, idx):
        # 1. 同时加载 imu + airpods 的 30s 段
        with h5py.File(self.data_path, 'r') as f:
            imu_30s = f['imu'][idx]              # (2048, 5, 6)
            air_30s = f['airpods'][idx] if self.use_airpods else None  # (2048, 9) or None

        # 2. 预处理
        imu_feat = self._preprocess_imu(imu_30s)    # [30, 2048]
        if self.use_airpods:
            air_feat = self._preprocess_airpods(air_30s)  # [6, 2048]
            sample_30s = torch.cat([imu_feat, air_feat], dim=0)  # [36, 2048]
        else:
            sample_30s = imu_feat                          # [30, 2048]

        # 3. 生成候选框
        proposal_boxes = generate_proposal_boxes(
            T_global=self.T_global,
            num_proposals=self.num_proposals
        )

        if self.merge_proposals:
            dummy_scores = torch.rand(self.num_proposals, 30)  # TODO
            proposal_boxes = merge_overlapping_proposals(proposal_boxes, dummy_scores)

        # 4. 视频级标签
        video_label = self._get_video_level_label(idx)

        # 调试输出（只打印一次）
        if self._debug_print:
            print("====== [WeakIMUDatasetTrain] sample debug ======")
            print(f"idx = {idx}")
            print(f"imu_30s raw shape        : {imu_30s.shape}")        # (2048, 5, 6)
            if self.use_airpods:
                print(f"air_30s raw shape        : {air_30s.shape}")    # (2048, 9)
                print(f"imu_feat shape           : {imu_feat.shape}")   # [30, 2048]
                print(f"air_feat shape           : {air_feat.shape}")   # [6, 2048]
            print(f"final sample_30s shape    : {sample_30s.shape}")    # [30 or 36, 2048]
            print(f"proposal_boxes shape      : {proposal_boxes.shape}")# [num_proposals, 2]
            print(f"video_label shape         : {video_label.shape}")   # [num_classes]
            print("imu_feat mean/std:", imu_feat.mean().item(), imu_feat.std().item())
            print("air_feat mean/std:", air_feat.mean().item(), air_feat.std().item())
            print("================================================")
            self._debug_print = False

        return sample_30s, proposal_boxes, video_label

class WWADLDatasetTestSingle():

    def __init__(self, config, modality=None, device_keep_list=None):
        self._debug_print = False  # True：只打印一次，False：不打印输出

        # 初始化路径配置
        self.dataset_dir = config['path']['test_dataset_path']
        dataset_root_path = config['path']['dataset_root_path']
        self.test_file_list = load_file_list(self.dataset_dir)

        # 读取info.json文件
        self.info_path = os.path.join(self.dataset_dir, "info.json")
        with open(self.info_path, 'r') as json_file:
            self.info = json.load(json_file)

        if modality is None:
            assert len(self.info['modality_list']) == 1, "single modality"
            self.modality = self.info['modality_list'][0]
        else:
            self.modality = modality
        # 构建测试文件路径列表
        if self.modality == 'airpods':
            self.file_path_list = [
                os.path.join(dataset_root_path, 'AirPodsPro', t)
                for t in self.test_file_list
            ]
        else:
            self.file_path_list = [
                os.path.join(dataset_root_path, self.modality, t)
                for t in self.test_file_list
            ]

        # 设置接收器过滤规则和新映射
        self.device_keep_list = device_keep_list
        print(f"device_keep_list: {self.device_keep_list}")
        self.new_mapping = self.info['segment_info'].get('new_mapping', None)

        # 定义模态数据集映射
        self.modality_dataset_map = {
            'imu': WWADL_imu,
            'airpods': WWADL_airpods
        }
        self.modality_dataset = self.modality_dataset_map[self.modality]

        # 加载分段和目标信息
        segment_info = self.info['segment_info']['train']
        self.clip_length = segment_info[modality]['window_len']
        self.stride = segment_info[modality]['window_step']
        self.target_len = self.info['segment_info']['target_len']

        # 加载评估标签路径
        self.eval_gt = os.path.join(self.dataset_dir, f'{self.modality}_annotations.json')

        # 加载全局均值和标准差
        self.global_mean, self.global_std = self.load_global_stats()

        # 初始化动作ID到动作映射
        self.id_to_action = self.info['segment_info'].get('id2action', {} )

        self.normalize = True

    def load_global_stats(self):
        """
        从文件加载全局均值和方差。
        如果文件中不存在当前 modality，则计算并更新文件。
        """
        stats_path = os.path.join(self.dataset_dir, "global_stats.json")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Global stats file '{stats_path}' not found. Ensure it is generated during training.")

        with open(stats_path, 'r') as f:
            stats = json.load(f)
        # 如果当前 modality 不在文件中，计算并保存
        if self.modality not in stats:
            raise FileNotFoundError(
                f"Modality '{self.modality}' not found in stats file. Ensure it is generated during training.")

        # 从文件中加载当前 modality 的均值和方差
        self.global_mean = np.array(stats[self.modality]["global_mean"])
        self.global_std = np.array(stats[self.modality]["global_std"])

        return self.global_mean, self.global_std

    def get_data(self, file_path):
        sample = self.modality_dataset(file_path,
                                       receivers_to_keep=None,
                                       new_mapping=self.new_mapping)
        sample_count = len(sample.data)

        # 生成 offset 列表，用于分割视频片段
        if sample_count < self.clip_length:
            offsetlist = [0]  # 视频长度不足 clip_length，只取一个片段
        else:
            offsetlist = list(range(0, sample_count - self.clip_length + 1, self.stride))  # 根据步长划分片段
            if (sample_count - self.clip_length) % self.stride:
                offsetlist += [sample_count - self.clip_length]  # 确保最后一个片段不被遗漏

        for offset in offsetlist:
            clip = sample.data[offset: offset + self.clip_length]  # 获取当前的 clip

            # 调用封装的函数进行处理
            clip = handle_nan_and_interpolate(clip, self.clip_length, self.target_len)
            assert not np.any(np.isnan(clip)), "Data contains NaN values!"

            clip = torch.tensor(clip, dtype=torch.float32)

            # 根据模态处理数据
            if self.modality == 'imu':
                clip = self.process_imu(clip)
            elif self.modality == 'airpods':
                clip = self.process_airpods(clip)

            data = {
                self.modality: clip
            }

            yield data, [offset, offset + self.clip_length]

    def process_imu(self, sample):
        sample = sample.permute(1, 2, 0)  # [5, 6, 2048]
        device_num = sample.shape[0]
        imu_channel = sample.shape[1]
        sample = sample.reshape(-1, sample.shape[-1])  # [5*6=30, 2048]
        # 全局归一化：使用序列维度的均值和标准差
        if self.normalize:
            sample = (sample - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
                     (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)

        if self.device_keep_list:
            # 1. sample [5*6=30, 2048] -> [5, 6, 2048]
            sample = sample.reshape(device_num, imu_channel, -1)  # [5, 6, 2048]
            # 2. 保留设备 [5, 6, 2048] -> [2, 6, 2048] 保留设备 2, 3
            sample = sample[self.device_keep_list]  # [len(device_keep_list), 6, 2048]
            # 3. [2, 6, 2048] -> [2*6=12, 2048]
            sample = sample.reshape(-1, sample.shape[-1])  # [5*6=30, 2048]

        return sample

    def process_airpods(self, sample):
        # AirPods 数据处理：处理加速度和角速度
        acceleration = sample[:, 3:6]  # 加速度: X, Y, Z (列索引 3 到 5)
        rotation = sample[:, 6:9]  # 角速度: X, Y, Z (列索引 6 到 8)

        # 合并加速度和角速度数据 [2048, 6]
        sample = torch.cat((acceleration, rotation), dim=1)  # [2048, 6]

        # 转置为 [6, 2048]
        sample = sample.T  # 转置为 [6, 2048]

        # 全局归一化：使用序列维度的均值和标准差
        if self.normalize:
            sample = (sample - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
                     (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)
        return sample

    def dataset(self):
        for file_path, file_name in zip(self.file_path_list, self.test_file_list):
            yield file_name, self.get_data(file_path)

class WeaklySupervisedXRFV2DatasetTest:
    def __init__(self,
                 config,
                 modality: str = 'imu',
                 device_keep_list=None,
                 use_airpods: bool = False):
        assert modality == 'imu', "WeaklySupervisedXRFV2DatasetTest 目前只支持 imu 作为主模态"
        self.modality = modality
        self._debug_print = False

        # 1) IMU 测试数据集
        self.imu_dataset = WWADLDatasetTestSingle(
            config,
            modality='imu',
            device_keep_list=device_keep_list
        )

        # 2) 是否启用 AirPods —— 用传进来的参数
        self.use_airpods = use_airpods
        if self.use_airpods:
            self.airpods_dataset = WWADLDatasetTestSingle(
                config,
                modality='airpods',
                device_keep_list=None
            )
        else:
            self.airpods_dataset = None

        # 3) 评估仍然用 IMU 的 GT
        self.clip_length = self.imu_dataset.clip_length
        self.stride = self.imu_dataset.stride
        self.eval_gt = self.imu_dataset.eval_gt
        self.id_to_action = self.imu_dataset.id_to_action
        self.normalize = True

    def dataset(self):
        # 情况一：只用 IMU（不拼 AirPods）
        if not self.use_airpods or self.airpods_dataset is None:
            for file_name, imu_iter in self.imu_dataset.dataset():
                # ★ 调试：只看第一个文件的第一段
                if self._debug_print:
                    print("====== [WeakIMUDatasetTest] IMU only debug ======")
                    print(f"file_name: {file_name}")
                    # imu_iter 是一个 generator，我们手动取一段看看
                    first_clip, first_seg = next(imu_iter)
                    print(f"first segment index range: {first_seg}")
                    print(f"first clip keys          : {first_clip.keys()}")
                    print(f"first clip['imu'].shape  : {first_clip['imu'].shape}")  # [30, T]
                    print("==================================================")
                    self._debug_print = False

                    # 把刚刚消耗的这个 clip 再包回一个新的 generator
                    def regen():
                        yield first_clip, first_seg
                        for item in imu_iter:
                            yield item
                    yield file_name, regen()
                else:
                    yield file_name, imu_iter
        else:
            imu_files = self.imu_dataset.dataset()
            air_files = self.airpods_dataset.dataset()

            for (imu_name, imu_iter), (air_name, air_iter) in zip(imu_files, air_files):
                assert imu_name == air_name, f"IMU/AirPods 文件名不一致: {imu_name} vs {air_name}"

                def merged_iter(imu_iter=imu_iter, air_iter=air_iter):
                    for (imu_clip, seg_imu), (air_clip, seg_air) in zip(imu_iter, air_iter):
                        imu_feat = imu_clip['imu']
                        air_feat = air_clip['airpods']
                        merged = torch.cat([imu_feat, air_feat], dim=0)

                        # # ★ 只在第一段打印一次
                        # nonlocal_first = getattr(self, "_debug_print", True)
                        # if nonlocal_first:
                        #     print("====== [WeakIMUDatasetTest] IMU + AirPods debug ======")
                        #     print(f"file_name: {imu_name}")
                        #     print(f"segment index range     : {seg_imu}")  # [0,1500]
                        #     print(f"imu_feat.shape          : {imu_feat.shape}")    # [30,2048],[30, T]
                        #     print(f"air_feat.shape          : {air_feat.shape}")    # [6,2048],[6, T]
                        #     print(f"merged imu shape        : {merged.shape}")      # [36, T]
                        #     print("======================================================")
                        #     self._debug_print = False

                        yield {'imu': merged}, seg_imu

                yield imu_name, merged_iter()

class WWADLBase:
    def __init__(self, file_path):
        self.data = None
        self.label = None
        self.file_name = os.path.basename(file_path)
        self.window_len = 0
        self.window_step = 0

    def load_data(self, file_path):
        pass

    def mapping_label(self, old_to_new_mapping):
        for i in range(len(self.label)):
            try:
                self.label[i][1] = old_to_new_mapping[str(self.label[i][1])]
            except:
                print(self.label[i][1], old_to_new_mapping)

class WWADL_imu(WWADLBase):
    """
    数据维度说明:
    - 数据 shape: (2900, 5, 6)
        - 第一个维度 (2900): 样本数量（例如时间序列的时间步）
        - 第二个维度 (5): 设备数量（5个IMU设备，对应位置见 name_to_id）
        - 第三个维度 (6): IMU数据的维度（例如加速度和陀螺仪的6个轴数据）

    name_to_id 映射设备位置到索引:
        - 'glasses': 0
        - 'left hand': 1
        - 'right hand': 2
        - 'left pocket': 3
        - 'right pocket': 4
    """
    def __init__(self, file_path, receivers_to_keep=None, new_mapping=None):
        """
        初始化 IMU 数据处理类，并保留指定设备的维度
        Args:
            file_path (str): 数据文件路径
            devices_to_keep (list, optional): 要保留的设备名称列表（如 ['glasses', 'left hand']）。
        """
        super().__init__(file_path)
        self.duration = 0
        self.load_data(file_path)

        # 如果提供了需要保留的设备列表，则过滤设备维度
        if receivers_to_keep:
            self.retain_devices(receivers_to_keep)

        if new_mapping:
            self.mapping_label(new_mapping)

    def load_data(self, file_path):
        """
        加载IMU数据并完成预处理

        Args:
            file_path (str): 数据文件的路径
        """
        data = load_h5(file_path)
        # 数据转置: (5, 2900, 6) 调整为 (2900, 5, 6)
        self.data = np.transpose(data['data'], (1, 0, 2))

        # 加载标签和持续时间
        self.label = data['label']
        self.duration = data['duration']

    def retain_devices(self, devices_to_keep):
        """
        过滤并保留指定设备的维度

        Args:
            devices_to_keep (list): 需要保留的设备名称列表
        """
        # 定义 name_to_id 映射
        name_to_id = {
            'glasses': 0,
            'left hand': 1,
            'right hand': 2,
            'left pocket': 3,
            'right pocket': 4,
        }

        # 根据名称映射找到对应的索引
        device_indices = [name_to_id[device] for device in devices_to_keep if device in name_to_id]

        # 保留指定设备维度
        self.data = self.data[:, device_indices, :]

class WWADL_airpods(WWADLBase):
    def __init__(self, file_path, receivers_to_keep = None, new_mapping=None):
        super().__init__(file_path)
        self.duration = 0
        self.load_data(file_path)
        if new_mapping:
            self.mapping_label(new_mapping)


    def load_data(self, file_path):
        data = load_h5(file_path)

        self.data = data['data']
        self.label = data['label']
        self.duration = data['duration']

class FullBackboneWrapper1D(nn.Module):
    """
    把只能吃固定长度 window 的 backbone，扩展到支持整条序列：
    - 滑窗跑 backbone 得到每段特征 f: [B,D,Lw]
    - 根据有效 stride(frames_per_bin) 把每段 f 回填到全局 feature 序列上
    - 重叠区域做平均（更稳），最后得到 global_feat: [B,D,T_global]
    """
    def __init__(self, backbone: nn.Module, win_len: int, stride: int, in_channels: int, probe_delta: int = None):
        super().__init__()
        self.backbone = backbone
        self.win_len = int(win_len)
        self.stride = int(stride)
        self.in_channels = int(in_channels)
        self.probe_delta = int(probe_delta) if probe_delta is not None else int(stride)

        self._calibrated = False
        self.frames_per_bin = None   # 有效下采样：raw_frames per 1 feature bin

    @torch.no_grad()
    def recalibrate(self, device, dtype):
        """
        用“长度增加 -> 输出长度增加”的差分法估计有效 stride。
        对 CNN/Conv 堆叠非常稳：frames_per_bin ≈ delta_raw / delta_L
        """
        T0 = self.win_len
        x0 = torch.zeros(1, self.in_channels, T0, device=device, dtype=dtype)
        y0 = self.backbone(x0)
        L0 = int(y0.shape[-1])

        delta = self.probe_delta
        for _ in range(8):
            x1 = torch.zeros(1, self.in_channels, T0 + delta, device=device, dtype=dtype)
            y1 = self.backbone(x1)
            L1 = int(y1.shape[-1])
            dL = L1 - L0
            if dL > 0:
                self.frames_per_bin = delta / float(dL)
                self._calibrated = True
                return
            delta *= 2

        raise RuntimeError("Failed to calibrate frames_per_bin (output length did not increase).")

    def _maybe_calibrate(self, x: torch.Tensor):
        if not self._calibrated:
            self.recalibrate(device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_info: bool = False):
        """
        x: [B,C,T]
        return: global_feat [B,D,T_global], info
        """
        B, C, T = x.shape
        if C != self.in_channels:
            raise ValueError(f"in_channels mismatch: x has {C}, wrapper expects {self.in_channels}")

        self._maybe_calibrate(x)

        # short -> direct
        if T <= self.win_len:
            feat = self.backbone(x)  # [B,D,L]
            info = {
                "raw_frames": T,
                "T_global": int(feat.shape[-1]),
                "win_len": self.win_len,
                "stride": self.stride,
                "frames_per_bin": float(self.frames_per_bin),
                "offsets": [0],
                "start_bins": [0],
            }
            return (feat, info) if return_info else feat

        # offsets（保证最后覆盖末尾）
        offsets = list(range(0, T - self.win_len + 1, self.stride))
        last = T - self.win_len
        if offsets[-1] != last:
            offsets.append(last)

        feats = []
        start_bins = []
        lens = []
        for s0 in offsets:
            clip = x[:, :, s0:s0 + self.win_len]
            f = self.backbone(clip)  # [B,D,Lw]
            sb = int(round(s0 / float(self.frames_per_bin)))  # 关键：window 起点对应的全局 bin 起点
            feats.append(f)
            start_bins.append(sb)
            lens.append(int(f.shape[-1]))

        # 估计全局长度
        T_global = max(sb + L for sb, L in zip(start_bins, lens))
        D = int(feats[0].shape[1])

        acc = torch.zeros(B, D, T_global, device=x.device, dtype=feats[0].dtype)
        cnt = torch.zeros(1, 1, T_global, device=x.device, dtype=feats[0].dtype)

        for f, sb, L in zip(feats, start_bins, lens):
            acc[:, :, sb:sb + L] += f
            cnt[:, :, sb:sb + L] += 1.0

        global_feat = acc / cnt.clamp(min=1.0)

        info = {
            "raw_frames": T,
            "T_global": int(global_feat.shape[-1]),
            "win_len": self.win_len,
            "stride": self.stride,
            "frames_per_bin": float(self.frames_per_bin),
            "offsets": offsets,
            "start_bins": start_bins,
            "lens": lens,
        }
        return (global_feat, info) if return_info else global_feat

def load_h5(filepath):
    def recursively_load_group_to_dict(h5file, path):
        """
        递归加载 HDF5 文件中的组和数据集为嵌套字典
        """
        result = {}
        group = h5file[path]

        for key, item in group.items():
            if isinstance(item, h5py.Group):
                # 如果是组，则递归加载
                result[key] = recursively_load_group_to_dict(h5file, f"{path}/{key}")
            elif isinstance(item, h5py.Dataset):
                # 如果是数据集，则加载为 NumPy 数组
                result[key] = item[()]

        return result

    with h5py.File(filepath, 'r') as h5file:
        return recursively_load_group_to_dict(h5file, '/')

def handle_nan_and_interpolate(data, window_len, target_len):
    """
    插值并在插值前处理 NaN 值的通用函数。
    Args:
        data (np.ndarray): 输入数据，维度为 (window_len, ...)
        window_len (int): 原始时序长度。
        target_len (int): 目标时序长度。
    Returns:
        np.ndarray: 插值后的数据，时间维度变为 target_len，其他维度保持不变。
    """
    original_shape = data.shape  # 获取原始形状
    flattened_data = data.reshape(window_len, -1)  # 展平除时间维度以外的所有维度

    # 定义插值函数
    def interpolate_channel(channel_data):
        original_indices = np.linspace(0, window_len - 1, window_len)
        target_indices = np.linspace(0, window_len - 1, target_len)

        # 检查 NaN 并处理
        nan_mask = np.isnan(channel_data)
        if np.any(nan_mask):  # 如果存在 NaN 值
            valid_indices = np.where(~nan_mask)[0]
            valid_values = channel_data[~nan_mask]

            if len(valid_indices) > 1:  # 至少两个有效值
                interp_func = interp1d(valid_indices, valid_values, kind='linear', bounds_error=False, fill_value="extrapolate")
                channel_data = interp_func(np.arange(window_len))
            else:
                # 如果有效值不足，填充为 0
                channel_data = np.zeros_like(channel_data)

        # 插值到目标长度
        interp_func = interp1d(original_indices, channel_data, kind='linear', bounds_error=False, fill_value="extrapolate")
        return interp_func(target_indices)

    # 对所有通道进行插值处理
    interpolated_flattened_data = np.array([
        interpolate_channel(flattened_data[:, i])
        for i in range(flattened_data.shape[1])
    ]).T  # 转置回时间维度在前

    # 恢复原始形状，时间维度替换为 target_len
    reshaped_interpolated_data = interpolated_flattened_data.reshape(target_len, *original_shape[1:])
    return reshaped_interpolated_data

def load_file_list(dataset_path):
    # 读取 test.csv
    test_csv_path = os.path.join(dataset_path, 'test.csv')
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"{test_csv_path} does not exist.")

    print("Loading test.csv...")
    test_df = pd.read_csv(test_csv_path)
    file_name_list = test_df['file_name'].tolist()
    print(f"Loaded {len(file_name_list)} file names from test.csv.")

    return file_name_list


if __name__ == "__main__":
    config = {
        "path": {
            "train_dataset_path": "/home/lipei/XRFV2/",
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "/home/lipei/project/WSDDN/checkpoints/",
            "result_path": "/home/lipei/project/WSDDN/test_results/"
        },
    }

    # ==== 1) 训练集检查 ====
    ds_train_imu = WeaklySupervisedXRFV2DatasetTrain(
        dataset_dir=config["path"]["train_dataset_path"],
        mapping_path=config["path"]["mapping_path"],
        use_airpods=False,
    )
    x, prop, y = ds_train_imu[0]
    print("[Train] IMU only sample_30s:", x.shape)

    ds_train_imu_air = WeaklySupervisedXRFV2DatasetTrain(
        dataset_dir=config["path"]["train_dataset_path"],
        mapping_path=config["path"]["mapping_path"],
        use_airpods=True,
    )
    x2, prop2, y2 = ds_train_imu_air[0]
    print("[Train] IMU + AirPods sample_30s:", x2.shape)

    # ==== 2) 测试集检查 ====
    test_ds = WeaklySupervisedXRFV2DatasetTest(
        config=config,
        modality='imu',
        device_keep_list=None,
        use_airpods=False,
    )
    for fname, data_iter in test_ds.dataset():
        clip_dict, seg = next(data_iter)
        print("[Test] first file:", fname)
        print("      segment range:", seg)
        print("      clip['imu'].shape:", clip_dict['imu'].shape)  # 预期: [36, T]
        break