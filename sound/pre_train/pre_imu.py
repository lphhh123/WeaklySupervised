# pre_imu.py
import copy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np
import h5py
from tqdm import tqdm
import json
import os


from .pre_model import *
from .pre_tsse_mamba_model_7s import TSSEClassifier_7s,TSSEMambaClassifier_7s

# -------------------------- 1. 全局映射配置（核心修改） --------------------------
# 1.1 旧ID → 英文label（你的id_to_action，key为字符串旧ID）
old_id_to_label = {
    '0': 'Stretching',
    '1': 'Pouring Water',
    '2': 'Writing',
    '3': 'Cutting Fruit',
    '4': 'Eating Fruit',
    '5': 'Taking Medicine',
    '6': 'Drinking Water',
    '7': 'Sitting Down',
    '8': 'Turning On/Off Eye Protection Lamp',
    '9': 'Opening/Closing Curtains',
    '10': 'Opening/Closing Windows',
    '11': 'Typing',
    '12': 'Opening Envelope',
    '13': 'Throwing Garbage',
    '14': 'Picking Fruit',
    '15': 'Picking Up Items',
    '16': 'Answering Phone',
    '17': 'Using Mouse',
    '18': 'Wiping Table',
    '19': 'Writing on Blackboard',
    '20': 'Washing Hands',
    '21': 'Using Phone',
    '22': 'Reading',
    '23': 'Watering Plants',
    '24': 'Walking to Bed',
    '25': 'Walking to Chair',
    '26': 'Walking to Cabinet',
    '27': 'Walking to Window',
    '28': 'Walking to Blackboard',
    '29': 'Getting Out of Bed',
    '30': 'Standing Up',
    '31': 'Lying Down',
    '32': 'Standing Still',
    '33': 'Lying Still'
}

# 1.2 旧ID → 新ID（你的Old to New Mapping，key为整数旧ID）
old_to_new_id = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19,
    20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 24, 26: 24, 27: 24, 28: 24,
    29: 25, 30: 26, 31: 27, 32: 28, 33: 29
}

# 1.3 英文label → 旧ID（反向映射，用于从标注文件的label找到旧ID）
label_to_old_id = {v: int(k) for k, v in old_id_to_label.items()}

class XRFV2Dataset_7s(Dataset):
    def __init__(self, dataset_dir, split="train", segment_len_7s=478, stride_7s=239, normalize=True, modality='imu', device_keep_list=None,use_airpods=False):
        """
        完全对齐WWADLDatasetSingle的初始化逻辑，仅新增7s子片段切分
        :param dataset_dir: 数据集根目录
        :param split: 仅支持"train"
        :param segment_len_7s: 7s子片段时间维度（478）
        :param stride_7s: 7s子片段滑动步长（239）
        """
        assert split == "train", "预训练仅支持训练集"
        self.dataset_dir = dataset_dir
        self.split = split
        self.normalize = normalize
        self.modality = modality
        self.device_keep_list = device_keep_list
        self.segment_len_7s = segment_len_7s
        self.stride_7s = stride_7s
        self.total_len_30s = 2048  # 30s样本固定时间维度

        self.use_airpods = use_airpods # 使用airpods


        self.data_path = os.path.join(dataset_dir, f"{split}_data.h5")
        self.label_path = os.path.join(dataset_dir, f"{split}_label.json")
        self.info_path = os.path.join(dataset_dir, "info.json")
        self.stats_path = os.path.join(dataset_dir, "global_stats.json")

        # 加载info.json
        with open(self.info_path, 'r') as f:
            self.info = json.load(f)

        # 加载训练集label.json
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)[self.modality]  # {样本索引str: 锚框标签列表}

        # 加载/计算全局归一化参数
        if self.normalize:
            if not os.path.exists(self.stats_path):
                raise FileNotFoundError(
                    f"未找到 {self.stats_path}，请先用原脚本生成 global_stats.json（包含 imu 和 airpods）"
                )
            self._load_all_stats()

        # 预生成所有7s子片段的索引映射（30s样本索引 → 7s子片段起始位置）
        self.subseg_map = []
        self.num_30s_samples = self.__len_30s_samples()  # 30s样本总数（9660）
        for sample_idx in range(self.num_30s_samples):
            # 对每个30s样本，滑动切分7s子片段
            start_positions = list(range(0, self.total_len_30s - self.segment_len_7s + 1, self.stride_7s))
            # 补充最后一个子片段，避免尾部遗漏
            if start_positions and start_positions[-1] != self.total_len_30s - self.segment_len_7s:
                start_positions.append(self.total_len_30s - self.segment_len_7s)
            elif not start_positions:  # 极端情况：样本长度不足7s
                start_positions = [0]
            # 记录映射（样本索引，7s子片段起始位置）
            for start in start_positions:
                self.subseg_map.append((sample_idx, start))

    def __len_30s_samples(self):
        with h5py.File(self.data_path, 'r') as f:
            return f[self.modality].shape[0]

    def compute_global_mean_std(self):
        """均值方差"""
        print("Calculating global mean and std...")
        mean_list, std_list = [], []
        with h5py.File(self.data_path, 'r') as f:
            data = f[self.modality]
            for i in tqdm(range(data.shape[0]), desc="Processing samples"):
                sample = data[i]
                if self.modality == 'imu':
                    sample = sample.transpose(1, 2, 0).reshape(-1, sample.shape[0])  # [30, 2048]
                # WiFi/airpods的处理逻辑可按需添加（同WWADLDatasetSingle）
                sample = torch.tensor(sample, dtype=torch.float32)
                mean_list.append(sample.mean(dim=1).numpy())
                std_list.append(sample.std(dim=1).numpy())
        global_mean = np.mean(mean_list, axis=0)
        global_std = np.mean(std_list, axis=0)
        return global_mean, global_std

    def save_global_stats(self):
        stats = {self.modality: {"global_mean": self.global_mean.tolist(), "global_std": self.global_std.tolist()}}
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f)
        print(f"Global stats saved to {self.stats_path}")

    def _load_all_stats(self):
        with open(self.stats_path, "r") as f:
            stats = json.load(f)

        # imu 统计量
        if "imu" not in stats:
            raise KeyError("global_stats.json 中没有 'imu' 的统计量")
        imu_stat = stats["imu"]
        self.global_mean_imu = np.array(imu_stat["global_mean"])
        self.global_std_imu = np.array(imu_stat["global_std"])

        # airpods 统计量（只有在 use_airpods=True 时才需要存在）
        if self.use_airpods:
            if "airpods" not in stats:
                raise KeyError("use_airpods=True，但 global_stats.json 中没有 'airpods' 的统计量")
            air_stat = stats["airpods"]
            self.global_mean_air = np.array(air_stat["global_mean"])
            self.global_std_air = np.array(air_stat["global_std"])

    def process_imu(self, sample):
        """维度调整+归一化+设备筛选"""
        sample = sample.permute(1, 2, 0)  # [5, 6, 2048]
        device_num = sample.shape[0]
        imu_channel = sample.shape[1]
        sample = sample.reshape(-1, sample.shape[-1])  # [30, 2048]

        if self.normalize:
            mean = torch.tensor(self.global_mean_imu, dtype=torch.float32)[:, None]
            std = torch.tensor(self.global_std_imu, dtype=torch.float32)[:, None] + 1e-6
            sample = (sample - mean) / std

        if self.device_keep_list:
            sample = sample.reshape(device_num, imu_channel, -1)
            sample = sample[self.device_keep_list]
            sample = sample.reshape(-1, sample.shape[-1])

        return sample

    def process_airpods(self, sample):
        """
        AirPods: [2048, 9]，取加速度(3:6) + 角速度(6:9)，得到 [2048, 6] ， [6, 2048]，再归一化
        """
        acceleration = sample[:, 3:6]  # [2048, 3]
        rotation = sample[:, 6:9]  # [2048, 3]
        sample = torch.cat((acceleration, rotation), dim=-1)  # [2048, 6]
        sample = sample.T  # [6, 2048]

        if self.normalize:
            mean = torch.tensor(self.global_mean_air, dtype=torch.float32)[:, None]
            std = torch.tensor(self.global_std_air, dtype=torch.float32)[:, None] + 1e-6
            sample = (sample - mean) / std

        return sample  # [6, 2048]

    # -------------------------- 7s子片段核心逻辑 --------------------------
    def __len__(self):
        """返回7s子片段总数（9660个30s样本 × ~7个/样本 ≈ 67620）"""
        return len(self.subseg_map)

    def __getitem__(self, idx):
        # 1. 根据映射获取30s样本索引和7s子片段起始位置
        sample_idx, start = self.subseg_map[idx]

        # 2. 同时加载 imu + airpods 的 30s 数据
        with h5py.File(self.data_path, 'r') as f:
            imu_30s = torch.tensor(f['imu'][sample_idx], dtype=torch.float32)  # (2048, 5, 6)
            if self.use_airpods:
                air_30s = torch.tensor(f['airpods'][sample_idx], dtype=torch.float32)  # (2048, 9)
            else:
                air_30s = None

        # 3. 特征预处理
        imu_feat = self.process_imu(imu_30s)  # [30, 2048]
        if self.use_airpods and air_30s is not None:
            air_feat = self.process_airpods(air_30s)  # [6, 2048]
            sample_30s = torch.cat([imu_feat, air_feat], dim=0)  # [36, 2048]
        else:
            sample_30s = imu_feat  # [30, 2048]

        # 4. 切分7s子片段
        subseg_data = sample_30s[:, start:start + self.segment_len_7s]  # [C, 478]
        assert subseg_data.shape[1] == self.segment_len_7s, "7s子片段维度错误"

        ## 5. 解析30s样本的锚框标签，提取动作ID和对应的时间区间
        anchor_labels = self.labels[str(sample_idx)]  # 每个锚框：[左偏移, 右偏移, 旧动作ID]
        action_time_segments = []
        for anchor in anchor_labels:
            left_offset, right_offset, old_action_id = anchor[0], anchor[1], int(anchor[2])
            # 这里原来的写法其实就是把 offset 映射到 [0,30s]
            anchor_start_sec = left_offset * 30
            anchor_end_sec = right_offset * 30
            action_time_segments.append((old_action_id, [anchor_start_sec, anchor_end_sec]))

        # 6. 计算7s子片段的时间区间（秒）
        subseg_start_sec = (start / self.total_len_30s) * 30
        subseg_end_sec = ((start + self.segment_len_7s) / self.total_len_30s) * 30

        # 7. 匹配覆盖比例最大的动作，并映射为新ID
        max_overlap_ratio = 0.0
        best_new_id = 29  # 默认新ID（Lying Still）
        for old_id, (act_start, act_end) in action_time_segments:
            overlap_start = max(subseg_start_sec, act_start)
            overlap_end = min(subseg_end_sec, act_end)
            if overlap_end <= overlap_start:
                continue
            overlap_ratio = (overlap_end - overlap_start) / (subseg_end_sec - subseg_start_sec)
            if overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio
                best_new_id = old_to_new_id.get(old_id, 29)

        return subseg_data, best_new_id

# -------------------------- 3. 通用训练函数--------------------------

def train_pretrain_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=1e-3,
    save_path='pretrain_model_best.pth'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() if model.task == 'single' else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        # ---------------- 1) 训练 ----------------
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0

        for data, labels in tqdm(train_loader, desc=f'[Train] Epoch {epoch + 1}/{num_epochs}'):
            data = data.to(device)
            labels = labels.to(device)
            if model.task == 'single':
                labels = labels.long()

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * data.size(0)

            if model.task == 'single':
                preds = outputs.argmax(dim=1)
                total_train_acc += (preds == labels).sum().item()
            else:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_train_acc += (preds == labels).sum().item() / labels.size(1)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_acc = total_train_acc / len(train_loader.dataset)

        # ---------------- 2) 验证 ----------------
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0

        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f'[ Val ] Epoch {epoch + 1}/{num_epochs}'):
                data = data.to(device)
                labels = labels.to(device)
                if model.task == 'single':
                    labels = labels.long()

                outputs = model(data)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item() * data.size(0)

                if model.task == 'single':
                    preds = outputs.argmax(dim=1)
                    total_val_acc += (preds == labels).sum().item()
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    total_val_acc += (preds == labels).sum().item() / labels.size(1)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_acc = total_val_acc / len(val_loader.dataset)

        print(
            f'Epoch {epoch + 1}/{num_epochs} | '
            f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | '
            f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}'
        )

        # --------- 3) 根据 val_loss 保存最佳模型 ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.backbone.state_dict())
            torch.save(best_state, save_path)
            print(f'>>> Val Loss 降到 {best_val_loss:.4f}，已保存当前最佳模型到 {save_path}')

        scheduler.step()

    print(f'训练结束，最佳 Val Loss = {best_val_loss:.4f}，对应模型已保存在 {save_path}')
# -------------------------- 4. 预训练启动入口  --------------------------
if __name__ == '__main__':
    config = {
        'dataset_dir': '/home/lipei/XRFV2',  # 训练集根目录（包含train_data.h5、train_label.json、info.json）
        'num_classes': 30,
        'batch_size': 32,
        'num_epochs': 60,
        'task': 'single',
        'segment_len_7s': 478,  # 2048/30*7=477.86-->478
        'stride_7s': 239,  # 50%
        'modality': 'imu',
        'device_keep_list': None,  # 可指定保留的设备，如[2,3]
        'use_airpods': True
    }

    # 加载对齐后的7s预训练数据集
    dataset = XRFV2Dataset_7s(
        dataset_dir=config['dataset_dir'],
        split='train',
        segment_len_7s=config['segment_len_7s'],
        stride_7s=config['stride_7s'],
        normalize=True,
        modality=config['modality'],
        device_keep_list=config['device_keep_list'],
        use_airpods=config['use_airpods'],
    )
    # print("imu mean shape:", dataset.global_mean_imu.shape)
    # print("air mean shape:", getattr(dataset, "global_mean_air", None))

    sample_data, label = dataset[0]
    print("subseg_data shape:", sample_data.shape)  # 期望: [36, 478] (imu30 + airpods6)
    print("label:", label)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)  # 这里是 4:1 -> 0.8 / 0.2
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ------- 根据是否使用 AirPods 设置 in_channels -------
    in_channels = 30 + (6 if config['use_airpods'] else 0)

    # 选择预训练模型
    model = CNN1DClassifier(num_classes=config['num_classes'], task=config['task'],in_channels=in_channels,)
    # model = TCNClassifier_7s(num_classes=config['num_classes'], task=config['task'])
    # model = ResTCNClassifier_7s(num_classes=config['num_classes'], task=config['task'])
    # model = MSTCNClassifier_7s(num_classes=config['num_classes'], task=config['task'])
    # model = LSTMClassifier_7s(num_classes=config['num_classes'], task=config['task'],in_channels=in_channels,)
    # model = TransformerClassifier_7s(num_classes=config['num_classes'], task=config['task'])
    # model = ActionFormerClassifier_7s(num_classes=config['num_classes'], task=config['task'],in_channels=in_channels,)
    # model = MambaClassifier_7s(num_classes=config['num_classes'], task=config['task'],in_channels=in_channels,)
    # model = TSSEMambaClassifier_7s(
    #     num_classes=config["num_classes"],
    #     task=config["task"],
    #     in_channels=in_channels,
    #     input_length=config["segment_len_7s"],
    #     embed_type="TSSE",                    #注意该点
    #     tsse_layers=2,
    #     mamba_cfg={"layer": 4, "mamba_type": "dbm"},
    # )
    # model = TSSEClassifier_7s(
    #     num_classes=config["num_classes"],
    #     task=config["task"],
    #     in_channels=in_channels,
    #     input_length=config["segment_len_7s"],
    #     tsse_layers=2,
    # )

    # 启动预训练
    if in_channels==30:
        save_model_path = f'XRFV2_{model.__class__.__name__}_pretrain_best.pth'
        # save_model_path = f'XRFV2_mamba_pretrain_best.pth'
    else:
        save_model_path = f'XRFV2_all_{model.__class__.__name__}_pretrain_best.pth'
        # save_model_path = f'XRFV2_all_mamba_pretrain_best.pth'

    train_pretrain_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        save_path=save_model_path,
    )