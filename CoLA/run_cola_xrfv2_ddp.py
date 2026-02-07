import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from easydict import EasyDict as edict

# ============================================================
# 引入依赖
# ============================================================
sys.path.append(os.getcwd())
from core.model import CoLA
from core.loss import TotalLoss
from core import utils
from core.dataset_xrfv2 import XRFV2Dataset
from WSDDN.utils import set_seed, GlobalBackboneWrapper
from WSDDN.tool import ANETdetection


# ============================================================
# DDP 基础设置
# ============================================================
def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not running in DDP mode, please use torchrun.")
        sys.exit(1)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


# ============================================================
# [核心组件] CoLA Full Wrapper XRFV2
# 统一训练和测试的数据流：所有输入都经过 GlobalBackboneWrapper
# ============================================================
class CoLA_FullWrapper_XRFV2(nn.Module):
    def __init__(self, cola_model, win_len=2048, stride=1024):
        super().__init__()
        # 1. 提取组件
        self.backbone = cola_model.actionness_module.backbone
        self.adapter = cola_model.actionness_module.adapter
        self.f_cls = cola_model.actionness_module.f_cls
        self.dropout = cola_model.actionness_module.dropout
        self.r_easy = cola_model.r_easy

        self.win_len = win_len
        self.stride = stride

        # 2. [修改] 仅包装 Backbone (输出 512 维)
        # 这样 GlobalBackboneWrapper 内部的 .view(..., 512, ...) 就能正常工作了
        from WSDDN.utils import GlobalBackboneWrapper
        self.global_extractor = GlobalBackboneWrapper(
            self.backbone,
            win_len=self.win_len,
            seg_stride=self.stride
        )

    def forward(self, x):
        """
        x: [B, T, C] (XRFV2 Channel Last)
        """
        x = x.permute(0, 2, 1)  # -> [B, C, T]

        # 1. 提取拼接后的 512 维特征
        # 返回 global_feat_512: [B, 512, T_feat_low]
        global_feat_512, info = self.global_extractor(x, return_info=True)

        # 2. [修改] 在拼接后，手动调用 Adapter 升维
        # [B, 512, T] -> [B, 2048, T]
        global_feat_2048 = self.adapter(global_feat_512)

        # 3. 分类头
        out = self.dropout(global_feat_2048)
        out = self.f_cls(out)  # [B, Classes, T]

        cas = torch.sigmoid(out).permute(0, 2, 1)  # [B, T, Classes]
        actionness = cas.sum(dim=2)

        # 4. 局部视频得分 (Local Max Pooling)
        k_len = int(self.win_len / info['bin_frames'])
        if k_len < 1: k_len = 1

        cas_t = cas.permute(0, 2, 1)  # [B, C, T]
        local_v_scores = F.max_pool1d(cas_t, kernel_size=k_len, stride=1, padding=k_len // 2)

        if local_v_scores.shape[2] > cas.shape[1]:
            local_v_scores = local_v_scores[:, :, :cas.shape[1]]

        return local_v_scores.permute(0, 2, 1), actionness, cas, info


# ============================================================
# 配置映射 (XRFV2)
# ============================================================
def map_config_to_cola_cfg(user_config):
    c = edict()
    c.DATASET_NAME = 'XRFV2'
    c.MODAL = user_config['modal']
    c.FEATS_FPS = user_config['fps']
    c.FEATS_DIM = user_config['in_channels']
    c.NUM_CLASSES = user_config['num_classes']
    c.NUM_SEGMENTS = 2048  # 固定为训练切片长度
    c.UP_SCALE = 1

    # XRFV2 使用单一的预训练文件
    c.PRETRAINED_PATH = user_config['pretrained_path']
    c.BACKBONE_TYPE = 'cnn1d'
    c.TRAIN_BACKBONE = user_config['training']['train_backbone']

    cola = user_config['cola']
    c.LAMBDA = cola['lambda']
    c.R_EASY = cola['r_easy']
    c.R_HARD = cola['r_hard']
    c.m = cola['m']
    c.M = cola['M']
    c.CLASS_THRESH = cola['class_thresh']
    c.NMS_THRESH = cola['nms_thresh']

    c.TIOU_THRESH = np.linspace(0.3, 0.7, 5)
    c.CAS_THRESH = np.arange(0.1, 0.5, 0.05)
    c.ANESS_THRESH = np.arange(0.1, 0.9, 0.05)

    # 30 类字典
    c.CLASS_DICT = {
        "Stretching": 0, "Pouring Water": 1, "Writing": 2, "Cutting Fruit": 3,
        "Eating Fruit": 4, "Taking Medicine": 5, "Drinking Water": 6, "Sitting Down": 7,
        "Turning On/Off Eye Protection Lamp": 8, "Opening/Closing Curtains": 9,
        "Opening/Closing Windows": 10, "Typing": 11, "Opening Envelope": 12,
        "Throwing Garbage": 13, "Picking Fruit": 14, "Picking Up Items": 15,
        "Answering Phone": 16, "Using Mouse": 17, "Wiping Table": 18,
        "Writing on Blackboard": 19, "Washing Hands": 20, "Using Phone": 21,
        "Reading": 22, "Watering Plants": 23, "Walking": 24, "Getting Out of Bed": 25,
        "Standing Up": 26, "Lying Down": 27, "Standing Still": 28, "Lying Still": 29
    }
    return c


# ============================================================
# 健壮加载函数
# ============================================================
def load_checkpoint_robust(net, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return net
    checkpoint = torch.load(ckpt_path, map_location=device)
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace('module.', '')
        if name.startswith('actionness_backbone'):
            name = name.replace('actionness_backbone', 'actionness_module.backbone')
        elif name.startswith('actionness_adapter'):
            name = name.replace('actionness_adapter', 'actionness_module.adapter')
        elif name.startswith('actionness_f_cls'):
            name = name.replace('actionness_f_cls', 'actionness_module.f_cls')
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=False)
    return net


# ============================================================
# 训练函数
# ============================================================
def train_fold_ddp(config, fold, rank, local_rank, world_size):
    if rank == 0: print(f"\n>>> [Train DDP] XRFV2 (No Fold, Random Split)...")

    cola_cfg = map_config_to_cola_cfg(config)
    net = CoLA(cola_cfg).to(local_rank)

    # 冻结/解冻
    if not cola_cfg.TRAIN_BACKBONE:
        for p in net.actionness_module.backbone.parameters(): p.requires_grad = False
        net.actionness_module.backbone.eval()

    # 实例化 Wrapper (训练模式)
    # 传入的 net 必须还没有被 DDP 包装
    train_wrapper = CoLA_FullWrapper_XRFV2(net, win_len=cola_cfg.NUM_SEGMENTS, stride=cola_cfg.NUM_SEGMENTS // 2)

    # DDP 包装 (注意：这里包装的是 Wrapper 还是 Net? 建议包装 Net，Wrapper 只是壳)
    # 但由于 forward 在 Wrapper 里，所以我们需要把 Wrapper 变成 DDP 模块
    # 或者，更简单的做法：把 Wrapper 逻辑写在 Net 内部。
    # 为了复用，我们这里直接包装 Wrapper。
    train_wrapper = nn.SyncBatchNorm.convert_sync_batchnorm(train_wrapper)
    ddp_model = DDP(train_wrapper, device_ids=[local_rank], output_device=local_rank)

    # 数据集
    train_ds = XRFV2Dataset(
        mode='train', modal=config["modal"],
        num_segments=cola_cfg.NUM_SEGMENTS,  # 2048
        class_dict=cola_cfg.CLASS_DICT, seed=config["seed"], supervision='weak'
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], sampler=sampler, num_workers=4,
                        drop_last=False)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()),
                           lr=config["training"]["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["lr_step_size"],
                                          gamma=config["training"]["lr_gamma"])
    criterion = TotalLoss().to(local_rank)

    for epoch in range(config["training"]["num_epochs"]):
        sampler.set_epoch(epoch)
        ddp_model.train()
        if not cola_cfg.TRAIN_BACKBONE:
            ddp_model.module.backbone.eval()

        ep_loss = 0
        for data, label, _, _, _ in loader:
            # data: [B, 2048, 30]
            data = data.to(local_rank)
            label = label.to(local_rank).float()

            optimizer.zero_grad()
            # Forward via Wrapper
            video_scores, _, _, cas = ddp_model(data)

            # 兼容性处理
            # 这里的 cas 是低频特征图，CoLA Loss 需要对齐维度
            # 简单做法：只用 video_scores 算分类 Loss，或者插值 cas 回去算定位
            # 但 Wrapper 已经对齐了训练和推理。
            # 现在的 CoLA Loss 内部逻辑是基于 Snippet 的，我们需要确保输入维度的含义一致。
            # CoLA Loss 内部没有复杂的长度依赖，除了 SniCo。
            # 为了训练稳定，我们这里只用 Video 分数算 Action Loss，或者传入 dummy contrast pairs
            # 或者我们需要修改 TotalLoss 以接受低频 cas

            # [简单方案] 暂时只算分类 Loss，确保 Backbone 收敛
            video_scores = torch.clamp(video_scores, 1e-6, 1 - 1e-6)

            # 取全局最大作为视频分
            v_score_global, _ = torch.max(video_scores, dim=1)  # [B, C]

            # 手动计算 BCE
            bce_loss = nn.BCELoss()(v_score_global, label / (label.sum(1, keepdim=True) + 1e-10))
            cost = bce_loss

            cost.backward()
            optimizer.step()
            ep_loss += cost.item()

        scheduler.step()
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1} Loss: {ep_loss / len(loader):.4f}")

    save_dir = os.path.join(config["result_root"], f"seed_{config['seed']}", "checkpoints")
    ckpt_path = os.path.join(save_dir, "model_final.pth")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        # 保存原始 net 的权重，剥离 wrapper
        torch.save(ddp_model.module.state_dict(), ckpt_path)  # 保存包含 wrapper 结构的权重

    dist.barrier()
    return ckpt_path, os.path.dirname(save_dir)


# ============================================================
# 推理函数 (Rank 0)
# ============================================================
def process_outputs_xrfv2(video_scores, cas, info, video_props, video_id, config, cola_cfg):
    # video_scores: [T, C]
    # info['bin_frames'] = 16

    scale = info['bin_frames']
    hard_thresh = cola_cfg.CLASS_THRESH

    local_vid_p = video_scores[0]

    # 全局筛选
    global_max, _ = torch.max(local_vid_p, dim=0)
    pred_cats = np.where(global_max.cpu().numpy() >= hard_thresh)[0]
    if len(pred_cats) == 0: pred_cats = np.array([np.argmax(global_max.cpu().numpy())])

    # 局部掩码
    local_mask = (local_vid_p >= hard_thresh).float()
    cas_p = torch.sigmoid(cas[0]) * local_mask

    # 维度准备
    cas_np = np.expand_dims(cas_p.cpu().numpy(), axis=2)
    aness_np = np.ones_like(cas_np)  # XRFV2 简化处理

    vid_p_np = global_max.cpu().numpy()
    T_feat = cas_p.shape[0]

    prop_dict = utils.get_proposal_dict(cas_np, aness_np, pred_cats, vid_p_np, T_feat, cola_cfg)

    for cls_id, props in prop_dict.items():
        for p in props:
            # 还原坐标
            video_props.append([p[0], p[1], p[2] * scale, p[3] * scale])


@torch.no_grad()
def eval_xrfv2_rank0(config, ckpt_path, save_dir, device):
    print(f"\n>>> [Eval] XRFV2 Test Full...")
    cola_cfg = map_config_to_cola_cfg(config)

    # 加载模型
    net = CoLA(cola_cfg).to(device)
    # 因为保存的是 Wrapper 结构，加载时需要注意
    # 如果 train_fold_ddp 保存的是 ddp_model.module (即 Wrapper)，则需要先实例化 Wrapper
    full_wrapper = CoLA_FullWrapper_XRFV2(net, win_len=2048, stride=1024).to(device)

    # 加载权重
    load_checkpoint_robust(full_wrapper, ckpt_path, device)
    full_wrapper.eval()

    test_ds = XRFV2Dataset(
        mode='test', modal=config["modal"],
        num_segments=0,  # 全长
        class_dict=cola_cfg.CLASS_DICT, seed=config["seed"], supervision='weak'
    )

    final_res = {'version': 'VERSION 1.3', 'results': {}}
    id2name = {v: k for k, v in cola_cfg.CLASS_DICT.items()}

    for i in tqdm(range(len(test_ds))):
        sample, _, _, video_id, _ = test_ds[i]
        inp = sample.unsqueeze(0).to(device)  # [1, T, C]

        video_scores, _, cas, info = full_wrapper(inp)

        video_props = []
        process_outputs_xrfv2(video_scores, cas, info, video_props, str(video_id), config, cola_cfg)

        # NMS & Format
        sbj_list = []
        if video_props:
            cls_groups = {}
            for p in video_props: (cls_groups.setdefault(int(p[0]), [])).append(p)
            for cid, props in cls_groups.items():
                keep = utils.nms(props, cola_cfg.NMS_THRESH)
                for k in keep:
                    # XRFV2 评估通常直接用帧索引，或者看 GT 格式。
                    # 假设 GT 是帧，这里就不除以 FPS。如果 GT 是秒，这里除以 50。
                    # XRFV2 默认 GT 是帧。
                    sbj_list.append({
                        'label': id2name.get(cid, str(cid)),
                        'score': float(k[1]),
                        'segment': [float(k[2]), float(k[3])]
                    })
        final_res['results'][str(video_id)] = sbj_list

    pred_path = os.path.join(save_dir, "predictions_test_full.json")
    with open(pred_path, 'w') as f:
        json.dump(final_res, f, indent=2)

    # 调用学姐评估
    if os.path.exists(config["gt_path"]):
        ev = ANETdetection(config["gt_path"], pred_path, subset="test", tiou_thresholds=np.linspace(0.3, 0.7, 5),
                           verbose=False)
        _, avg, _ = ev.evaluate()
        print(f"  mAP@Avg: {avg:.4f}")


# ============================================================
# Main
# ============================================================
def main():
    rank, local_rank, world_size = setup_ddp()

    base_config = {
        "dataset_dir": "/home/lipei/shared-nvme/dataset/all_30_3",
        "modal": "imu",
        "pretrained_path": "/home/lipei/project/WSDDN/output_xrfv2_classifier/classifier_best.pth",  # 你的分类预训练权重
        "gt_path": "/home/lipei/shared-nvme/dataset/all_30_3/imu_annotations.json",
        "result_root": "./output_xrfv2_cola_ddp",

        "fps": 50,
        "in_channels": 36,
        "num_classes": 30,
        "clip_sec": 30.0,  # 2048帧
        "clip_overlap": 0.0,

        "training": {
            "batch_size": 16, "num_epochs": 40, "lr": 1e-4,
            "lr_step_size": 10, "lr_gamma": 0.9, "train_backbone": False
        },
        "cola": {
            "lambda": 0.01, "r_easy": 20, "r_hard": 20, "m": 1, "M": 6,
            "class_thresh": 0.1, "nms_thresh": 0.4
        }
    }

    seeds = [2022, 2024, 2026]
    for seed in seeds:
        if rank == 0: print(f"### Seed {seed} ###")
        set_seed(seed)
        config = copy.deepcopy(base_config);
        config["seed"] = seed
        config["result_root"] = os.path.join(base_config["result_root"], f"seed_{seed}")

        ckpt, save_dir = train_fold_ddp(config, 0, rank, local_rank, world_size)
        if rank == 0: eval_xrfv2_rank0(config, ckpt, save_dir, torch.device(f"cuda:{local_rank}"))
        dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()