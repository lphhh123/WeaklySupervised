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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from easydict import EasyDict as edict

# from scipy.interpolate import interp1d # [修改 1] test_full 不再需要插值，删除引用

# ============================================================
# 引入 CoLA 核心与依赖
# ============================================================
sys.path.append(os.getcwd())
from core.model import CoLA
from core.loss import TotalLoss
from core import utils
from WSDDN.HANGTIME.dataset_hangtime_ws import WeaklyHangtimeDataset
from WSDDN.utils import set_seed, build_gt_for_anet


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
# [修改 2] 重写 CoLA_FullWrapper，对齐学姐的特征拼接逻辑
# ============================================================
class CoLA_FullWrapper(nn.Module):
    def __init__(self, cola_model, win_len=1500, stride=750):
        super().__init__()
        # 直接获取子模块，绕过 Actionness_Module.forward 中的强制插值
        self.backbone = cola_model.actionness_module.backbone
        self.adapter = cola_model.actionness_module.adapter
        self.f_cls = cola_model.actionness_module.f_cls
        self.dropout = cola_model.actionness_module.dropout

        # 引用 CoLA 的一些属性用于后处理
        self.r_easy = cola_model.r_easy
        self.get_video_cls_scores = cola_model.get_video_cls_scores

        self.win_len = win_len
        self.stride = stride

    @torch.no_grad()
    def forward(self, x):
        """
        x: [1, T_total, C] (原始长视频，未缩放)
        """
        # 1. 调整输入维度适配 Conv1d: [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
        B, C, T_total = x.shape

        # 2. 滑窗提取特征 (Backbone + Adapter)
        # 这里的 win_len 是 1500，对应训练时的窗口大小
        offsets = list(range(0, T_total - self.win_len + 1, self.stride))
        if not offsets or offsets[-1] != T_total - self.win_len:
            if T_total >= self.win_len:
                offsets.append(T_total - self.win_len)
            else:
                offsets = [0]
                # 补零 padding
                pad_len = self.win_len - T_total
                x = torch.nn.functional.pad(x, (0, pad_len))

        feat_list = []
        count_list = []

        # 预计算输出特征维度 (假设输入1500 -> 输出94)
        # 先跑一次 dummy 确定输出长度
        dummy_out = self.adapter(self.backbone(x[:, :, :self.win_len]))
        feat_dim = dummy_out.shape[1]  # 2048
        out_win_len = dummy_out.shape[2]  # 94 (取决于 backbone 下采样)

        # 计算采样率 (input_frames / feature_frames)
        # 例如 1500 / 94 ≈ 16
        rate = self.win_len / out_win_len

        # 计算全局特征图的长度
        T_out_total = int(T_total / rate) + 1

        global_feat = torch.zeros(B, feat_dim, T_out_total, device=x.device)
        count_map = torch.zeros(B, 1, T_out_total, device=x.device)

        for start in offsets:
            end = start + self.win_len
            chunk = x[:, :, start:end]  # [B, C, 1500]

            # [核心] 只跑特征提取，不跑分类，也不插值
            feat = self.backbone(chunk)
            feat = self.adapter(feat)  # [B, 2048, 94]

            # 映射回全局特征图的位置
            start_out = int(start / rate)
            end_out = start_out + out_win_len

            # 边界保护
            valid_w = min(end_out, T_out_total) - start_out

            if valid_w > 0:
                global_feat[:, :, start_out:start_out + valid_w] += feat[:, :, :valid_w]
                count_map[:, :, start_out:start_out + valid_w] += 1.0

        # 平均重叠区域
        global_feat /= count_map.clamp(min=1.0)

        # 3. 全局分类 (Head)
        # 输入: [B, 2048, T_out_total] (拼接后的长特征)

        # CoLA Logic (手动执行 model.py 后半部分)
        embeddings = global_feat.permute(0, 2, 1)  # [B, T, C]

        out = self.dropout(global_feat)
        out = self.f_cls(out)  # [B, Classes, T]

        cas = out.permute(0, 2, 1)  # [B, T, Classes]
        actionness = cas.sum(dim=2)  # [B, T]

        # 计算 Video-level Score
        k_easy = max(1, cas.shape[1] // self.r_easy)
        video_scores = self.get_video_cls_scores(cas, k_easy)

        return video_scores, actionness, cas


# ============================================================
# 辅助函数：配置映射
# ============================================================
def map_config_to_cola_cfg(user_config, fold):
    c = edict()
    c.DATASET_NAME = 'HANGTIME'
    c.MODAL = 'imu'
    c.FEATS_FPS = user_config['fps']
    c.FEATS_DIM = user_config['in_channels']
    c.NUM_CLASSES = user_config['num_classes']
    c.NUM_SEGMENTS = int(user_config['clip_sec'] * c.FEATS_FPS)  # 1500
    c.UP_SCALE = 1

    ckpt_name = f"hangtime_{user_config['pretrained_model_name']}_pretrained_loso_sbj_{fold}.pth"
    c.PRETRAINED_PATH = os.path.join(user_config['pretrained_dir'], ckpt_name)
    c.BACKBONE_TYPE = 'cnn1d'
    c.TRAIN_BACKBONE = user_config['training']['train_backbone']

    cola_cfg = user_config['cola']
    c.LAMBDA = cola_cfg['lambda']
    c.R_EASY = cola_cfg['r_easy']
    c.R_HARD = cola_cfg['r_hard']
    c.m = cola_cfg['m']
    c.M = cola_cfg['M']
    c.CLASS_THRESH = cola_cfg['class_thresh']
    c.NMS_THRESH = cola_cfg['nms_thresh']
    c.TIOU_THRESH = np.linspace(0.3, 0.7, 5)
    c.CAS_THRESH = np.arange(0.1, 0.4, 0.05)
    c.ANESS_THRESH = np.arange(0.1, 0.925, 0.025)
    c.CLASS_DICT = {"dribbling": 0, "shot": 1, "pass": 2, "rebound": 3, "layup": 4}
    return c


# ============================================================
# 推理辅助：滑窗与插值生成器
# ============================================================
def get_inference_data(dataset, mode, win_size=1500, stride=750):
    for sbj in dataset.subjects:
        raw = dataset._get_raw(sbj)
        if dataset.normalize:
            raw = (raw - dataset.mean) / (dataset.std + 1e-6)

        t_origin = raw.shape[0]

        if mode == "test_window":
            # [保持不变] 滑窗逻辑
            offsets = list(range(0, t_origin - win_size + 1, stride))
            if not offsets or offsets[-1] != t_origin - win_size:
                if t_origin >= win_size:
                    offsets.append(t_origin - win_size)
                else:
                    offsets = [0];
                    raw = np.pad(raw, ((0, win_size - t_origin), (0, 0)), mode='constant')

            def window_iter():
                for start in offsets:
                    yield torch.from_numpy(raw[start:start + win_size]).float().unsqueeze(0), start

            yield sbj, window_iter(), t_origin

        else:
            # [修改 3] test_full 逻辑：移除插值，直接返回原始数据
            # 原始逻辑是 interp1d，现在改为直接 yield raw
            # CoLA_FullWrapper 会处理 [1, T_raw, C] 的输入

            def single_iter():
                # [1, T_raw, C]
                yield torch.from_numpy(raw).float().unsqueeze(0), 0

            yield sbj, single_iter(), t_origin


# ============================================================
# 核心逻辑：Train One Fold (DDP)
# ============================================================
def train_one_fold_ddp(config, fold, rank, local_rank, world_size):
    cola_cfg = map_config_to_cola_cfg(config, fold)
    net = CoLA(cola_cfg).to(local_rank)

    if not cola_cfg.TRAIN_BACKBONE:
        for p in net.actionness_module.backbone.parameters(): p.requires_grad = False
        net.actionness_module.backbone.eval()

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    train_dataset = WeaklyHangtimeDataset(
        dataset_dir=config["dataset_dir"], loso_json=f"loso_sbj_{fold}.json",
        mode="train", fps=config["fps"], num_sensors=config["in_channels"],
        clip_sec=config["clip_sec"], clip_overlap=config["clip_overlap"],
        num_classes=config["num_classes"], normalize=True, seed=config["seed"]
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], sampler=sampler, num_workers=4,
                        drop_last=True)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config["training"]["lr"],
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["lr_step_size"],
                                          gamma=config["training"]["lr_gamma"])
    criterion = TotalLoss().to(local_rank)

    for epoch in range(config["training"]["num_epochs"]):
        sampler.set_epoch(epoch)
        net.train()
        if not cola_cfg.TRAIN_BACKBONE: net.module.actionness_module.backbone.eval()

        for data, label in loader:
            data = data.permute(0, 2, 1).to(local_rank)
            label = label.to(local_rank).float()

            optimizer.zero_grad()
            video_scores, contrast_pairs, _, _ = net(data)
            video_scores = torch.clamp(video_scores, min=1e-6, max=1.0 - 1e-6)
            cost, loss_dict = criterion(video_scores, label, contrast_pairs)

            if epoch < 10: cost = loss_dict['Loss/Action'] + 0.1 * loss_dict['Loss/SniCo']

            cost.backward()
            optimizer.step()
        scheduler.step()

    save_path = os.path.join(config["result_root"], f"fold{fold}", "model_final.pth")
    if rank == 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(net.module.state_dict(), save_path)
    dist.barrier()
    return save_path


# ============================================================
# 核心逻辑：Inference One Fold (Rank 0 only)
# ============================================================
@torch.no_grad()
def run_inference_dual_mode(config, fold, ckpt_path, device):
    # 1. 加载模型
    cola_cfg = map_config_to_cola_cfg(config, fold)
    net = CoLA(cola_cfg).to(device)

    # 兼容 DDP 权重的加载逻辑
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # [兼容性修复] 确保新旧 Key 匹配
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('actionness_backbone', 'actionness_module.backbone')
        k = k.replace('actionness_adapter', 'actionness_module.adapter')
        k = k.replace('actionness_f_cls', 'actionness_module.f_cls')
        new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    net.eval()

    # 2. 准备数据集
    loso_json = f"loso_sbj_{fold}.json"
    test_dataset = WeaklyHangtimeDataset(
        dataset_dir=config["dataset_dir"], loso_json=loso_json, mode="test_window",
        fps=config["fps"], num_sensors=config["in_channels"], clip_sec=config["clip_sec"],
        normalize=True
    )
    id2name = {v: k for k, v in cola_cfg.CLASS_DICT.items()}

    # 3. 初始化 Full Wrapper (用于 test_full)
    # 传入原始模型，Wrapper 内部会拆解
    full_wrapper = CoLA_FullWrapper(net, win_len=cola_cfg.NUM_SEGMENTS, stride=cola_cfg.NUM_SEGMENTS // 2)

    # 4. 双模式推理循环
    for mode in ["test_window", "test_full"]:
        final_res = {'version': 'VERSION 1.3', 'results': {}, 'external_data': {}}
        win_size = cola_cfg.NUM_SEGMENTS
        inf_time_list = []
        gpu_mem_list = []

        # [修改 4] 分支逻辑更新：test_full 使用 Wrapper
        if mode == "test_full":
            for sbj, window_iter, t_origin in get_inference_data(test_dataset, mode, win_size):
                # test_full 下 window_iter 只会 yield 一次 (whole sequence)
                for chunk, _ in window_iter:
                    chunk = chunk.to(device)
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    start_t = time.time()
                    video_scores, actionness, cas = full_wrapper(chunk.to(device))
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_t = time.time()
                    inf_time_list.append((end_t - start_t) * 1000.0)  # ms
                    if torch.cuda.is_available():
                        gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)  # MB
                    # 此时 T_current = T_feat (拼接后的特征长度)
                    T_feat = cas.shape[1]

                    # 坐标缩放因子: T_origin / T_feat
                    # 例如 3000 / 187 ≈ 16
                    scale = t_origin / T_feat

                    process_and_save(
                        sbj, video_scores, actionness, cas,
                        t_origin=t_origin,  # 原始长度
                        scale_factor=scale,  # [关键] 传入缩放因子
                        start_f=0, mode=mode, config=config, cola_cfg=cola_cfg,
                        id2name=id2name, final_res=final_res
                    )

        else:
            # test_window: 传统的滑窗
            stride = win_size // 2
            for sbj, window_iter, t_origin in get_inference_data(test_dataset, mode, win_size, stride):
                for chunk, start_f in window_iter:
                    chunk = chunk.to(device)
                    # 普通 Forward (内部有插值，但因为 chunk 是 1500，所以 1:1)
                    video_scores, _, actionness, cas = net(chunk)

                    process_and_save(
                        sbj, video_scores, actionness, cas,
                        t_origin=t_origin,
                        scale_factor=1.0,  # test_window 是 1:1 映射
                        start_f=start_f, mode=mode, config=config, cola_cfg=cola_cfg,
                        id2name=id2name, final_res=final_res
                    )

        # 保存
        if inf_time_list:
            avg_time = np.mean(inf_time_list)
            std_time = np.std(inf_time_list)
            avg_mem = np.mean(gpu_mem_list) if gpu_mem_list else 0.0
            std_mem = np.std(gpu_mem_list) if gpu_mem_list else 0.0

            stats_info = {
                "test_mode": mode,
                "num_samples": len(inf_time_list),
                "avg_inference_time_ms": f"{avg_time:.2f} ± {std_time:.2f}",
                "avg_gpu_memory_mb": f"{avg_mem:.2f} ± {std_mem:.2f}",
                "conf_thresh": config["cola"]["class_thresh"],  # 记录一下参数方便回溯
                "nms_thresh": config["cola"]["nms_thresh"]
            }

            # 保存路径与 predictions 同级
            stats_path = os.path.join(config["result_root"], f"fold{fold}", f"inference_stats_{mode}.json")
            with open(stats_path, 'w') as f:
                json.dump(stats_info, f, indent=2)

            print(f"    📊 [Stats {mode}] Time: {avg_time:.2f}ms | Mem: {avg_mem:.2f}MB")
        out_path = os.path.join(config["result_root"], f"fold{fold}", f"predictions_{mode}.json")
        with open(out_path, 'w') as f:
            json.dump(final_res, f, indent=2)
        print(f"    [Done] {mode} saved to {out_path}")


# ============================================================
# 辅助函数：统一的后处理逻辑
# ============================================================
def process_and_save(sbj, video_scores, actionness, cas, t_origin, scale_factor, start_f, mode, config, cola_cfg,
                     id2name,
                     final_res):
    # 1. Sigmoid & 融合
    cas_prob = torch.sigmoid(cas[0])  # [T, C]
    aness_prob = torch.sigmoid(actionness[0])  # [T]
    v_prob = torch.sigmoid(video_scores[0]).cpu().numpy()  # [C]

    # 融合：抑制背景
    cas_suppressed = cas_prob * aness_prob.unsqueeze(1)

    T_current = cas_prob.shape[0]

    cas_np = np.expand_dims(cas_suppressed.cpu().numpy(), axis=2)  # [T, C, 1]
    num_classes = cola_cfg.NUM_CLASSES
    aness_np = np.tile(aness_prob.cpu().numpy().reshape(-1, 1, 1), (1, num_classes, 1))

    # 类别筛选
    if mode == "test_full":
        cas_max_val, _ = torch.max(cas_prob, dim=0)
        cas_max_np = cas_max_val.cpu().numpy()
        pred_cats = np.where(cas_max_np >= cola_cfg.CLASS_THRESH)[0]
    else:
        pred_cats = np.where(v_prob >= cola_cfg.CLASS_THRESH)[0]

    if len(pred_cats) == 0:
        pred_cats = np.array([np.argmax(v_prob)])

    # 生成 Proposal (utils.py)
    # 注意：这里的 win_size 传入 T_current 即可，utils 内部用它做归一化
    prop_dict = utils.get_proposal_dict(cas_np, aness_np, pred_cats, v_prob, T_current, cola_cfg)

    video_props = []
    for cls_id, props in prop_dict.items():
        for p in props:
            # [修改 5] 坐标转换逻辑统一
            # p[2], p[3] 是基于 T_current (特征图长度) 的坐标
            # test_full: T_current = T_feat, scale = T_raw / T_feat
            # test_window: T_current = 1500, scale = 1.0 (因为我们没缩放), start_f 是偏移

            local_start = p[2] * scale_factor
            local_end = p[3] * scale_factor

            global_start = local_start + start_f
            global_end = local_end + start_f

            video_props.append([p[0], p[1], global_start, global_end])

    if sbj not in final_res['results']: final_res['results'][sbj] = []
    current_batch_list = []

    cls_groups = {}
    for p in video_props: (cls_groups.setdefault(int(p[0]), [])).append(p)

    for cid, props in cls_groups.items():
        keep = utils.nms(props, cola_cfg.NMS_THRESH)
        for k in keep:
            current_batch_list.append({
                'label': id2name.get(cid, str(cid)),
                'score': float(k[1]),
                'segment': [round(float(k[2]) / config["fps"], 2), round(float(k[3]) / config["fps"], 2)]
            })

    final_res['results'][sbj].extend(current_batch_list)
    final_res['results'][sbj] = sorted(
        final_res['results'][sbj],
        key=lambda x: x['score'],
        reverse=True
    )


# ============================================================
# Main Execution
# ============================================================
def main():
    rank, local_rank, world_size = setup_ddp()

    base_config = {
        "dataset_dir": "/home/lipei/TAL_data/hangtime/",
        "pretrained_dir": "/home/lipei/project/WSDDN/OtherData/HANGTIME/pre_train/CNN1D",
        "result_root": "./output_hangtime_cola_ddp",
        "fps": 50,
        "in_channels": 3,
        "num_classes": 5,
        "clip_sec": 30.0,
        "clip_overlap": 0.5,
        "folds": list(range(24)),
        "pretrained_model_name": "CNN1D",
        "training": {
            "batch_size": 16,  # Per-GPU batch
            "num_epochs": 80,
            "lr": 1e-4,
            "lr_step_size": 10,
            "lr_gamma": 0.9,
            "train_backbone": False
        },
        "cola": {
            "lambda": 0.01, "r_easy": 20, "r_hard": 20, "m": 3, "M": 10,
            "class_thresh": 0, "nms_thresh": 0.4
        }
    }

    seeds = [2022, 2024, 2026]

    for seed in seeds:
        if rank == 0: print(f"\n\n{'#' * 60}\n### SEED: {seed} ###\n{'#' * 60}")
        set_seed(seed)
        config = copy.deepcopy(base_config)
        config["seed"] = seed
        config["result_root"] = os.path.join(base_config["result_root"], f"seed_{seed}")

        for fold in config["folds"]:
            # 1. DDP Training
            ckpt_path = train_one_fold_ddp(config, fold, rank, local_rank, world_size)

            # 2. Sequential Inference (Rank 0 only)
            if rank == 0:
                device = torch.device(f"cuda:{local_rank}")
                run_inference_dual_mode(config, fold, ckpt_path, device)
            dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()