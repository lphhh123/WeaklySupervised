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
      
# ============================================================
sys.path.append(os.getcwd())
from core.model import CoLA
from core.loss import TotalLoss
from core import utils
                         
from WSDDN.Opportunity.dataset_opportunity_ws import WeaklyOpportunityDataset
from WSDDN.utils import set_seed, build_gt_for_anet
from WSDDN.tool import ANETdetection


# ============================================================
          
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
                                     
# ============================================================
class CoLA_FullWrapper(nn.Module):
    def __init__(self, cola_model, win_len, stride):
        super().__init__()
                                                     
        self.backbone = cola_model.actionness_module.backbone
        self.adapter = cola_model.actionness_module.adapter
        self.f_cls = cola_model.actionness_module.f_cls
        self.dropout = cola_model.actionness_module.dropout

                     
        self.r_easy = cola_model.r_easy
        self.get_video_cls_scores = cola_model.get_video_cls_scores

        self.win_len = win_len
        self.stride = stride

    @torch.no_grad()
    def forward(self, x):
                                                    
        x = x.permute(0, 2, 1)
        B, C, T_total = x.shape

                   
        offsets = list(range(0, T_total - self.win_len + 1, self.stride))
        if not offsets or offsets[-1] != T_total - self.win_len:
            if T_total >= self.win_len:
                offsets.append(T_total - self.win_len)
            else:
                offsets = [0]
                pad_len = self.win_len - T_total
                x = torch.nn.functional.pad(x, (0, pad_len))

                   
        dummy_out = self.adapter(self.backbone(x[:, :, :self.win_len]))
        feat_dim = dummy_out.shape[1]  # 2048
        out_win_len = dummy_out.shape[2]              

                                 
        rate = self.win_len / out_win_len
        T_out_total = int(T_total / rate) + 1

        global_feat = torch.zeros(B, feat_dim, T_out_total, device=x.device)
        count_map = torch.zeros(B, 1, T_out_total, device=x.device)

        for start in offsets:
            end = start + self.win_len
            chunk = x[:, :, start:end]

                    
            feat = self.backbone(chunk)
            feat = self.adapter(feat)

            start_out = int(start / rate)
            end_out = start_out + out_win_len
            valid_w = min(end_out, T_out_total) - start_out

            if valid_w > 0:
                global_feat[:, :, start_out:start_out + valid_w] += feat[:, :, :valid_w]
                count_map[:, :, start_out:start_out + valid_w] += 1.0

        global_feat /= count_map.clamp(min=1.0)

                 
        out = self.dropout(global_feat)
        out = self.f_cls(out)  # [B, Classes, T]

        cas = out.permute(0, 2, 1)  # [B, T, Classes]
        actionness = cas.sum(dim=2)  # [B, T]

        k_easy = max(1, cas.shape[1] // self.r_easy)
        video_scores = self.get_video_cls_scores(cas, k_easy)

        return video_scores, actionness, cas


# ============================================================
                          
# ============================================================
def map_config_to_cola_cfg(user_config, fold):
    c = edict()
    c.DATASET_NAME = 'Opportunity'
    c.MODAL = 'imu'
    c.FEATS_FPS = user_config['fps']  # 30
    c.FEATS_DIM = user_config['in_channels']  # 113
    c.NUM_CLASSES = user_config['num_classes']  # 17

                           
    c.NUM_SEGMENTS = int(user_config['clip_sec'] * c.FEATS_FPS)
    c.UP_SCALE = 1

                                
    ckpt_name = f"opportunity_{user_config['pretrained_model_name']}_pretrained_loso_sbj_{fold}.pth"
    c.PRETRAINED_PATH = os.path.join(user_config['pretrained_dir'], ckpt_name)
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

           
    c.CLASS_DICT = {
        "open_door_1": 0,
        "open_door_2": 1,
        "close_door_1": 2,
        "close_door_2": 3,
        "open_fridge": 4,
        "close_fridge": 5,
        "open_dishwasher": 6,
        "close_dishwasher": 7,
        "open_drawer_1": 8,
        "close_drawer_1": 9,
        "open_drawer_2": 10,
        "close_drawer_2": 11,
        "open_drawer_3": 12,
        "close_drawer_3": 13,
        "clean_table": 14,
        "drink_from_cup": 15,
        "toggle_switch": 16
    }
    return c


# ============================================================
         
# ============================================================
def get_inference_data(dataset, mode, win_size, stride):
    for sbj in dataset.subjects:
        raw = dataset._get_raw(sbj)
        if dataset.normalize:
            raw = (raw - dataset.mean) / (dataset.std + 1e-6)

        t_origin = raw.shape[0]

        if mode == "test_window":
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
                                                        
            def single_iter():
                yield torch.from_numpy(raw).float().unsqueeze(0), 0

            yield sbj, single_iter(), t_origin


# ============================================================
            
# ============================================================
def train_fold_ddp(config, fold, rank, local_rank, world_size):
    if rank == 0: print(f"\n>>> [Train DDP] Starting Fold {fold}...")

    cola_cfg = map_config_to_cola_cfg(config, fold)
    net = CoLA(cola_cfg).to(local_rank)

           
    if not cola_cfg.TRAIN_BACKBONE:
        for p in net.actionness_module.backbone.parameters(): p.requires_grad = False
        net.actionness_module.backbone.eval()

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    loso_json = f"loso_sbj_{fold}.json"
    train_ds = WeaklyOpportunityDataset(
        dataset_dir=config["dataset_dir"], loso_json=loso_json, mode="train",
        fps=config["fps"], num_sensors=config["in_channels"],
        clip_sec=config["clip_sec"], clip_overlap=config["clip_overlap"],
        num_classes=config["num_classes"], normalize=True, seed=config["seed"]
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], sampler=sampler, num_workers=4,
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

        ep_loss = 0
        for data, label in loader:
            data = data.permute(0, 2, 1).to(local_rank)
            label = label.to(local_rank).float()

            optimizer.zero_grad()
            video_scores, contrast_pairs, _, _ = net(data)
            video_scores = torch.clamp(torch.sigmoid(video_scores), 1e-6, 1 - 1e-6)

            cost, loss_dict = criterion(video_scores, label, contrast_pairs)

            # Warmup
            if epoch < 10: cost = loss_dict['Loss/Action'] + 0.1 * loss_dict['Loss/SniCo']

            cost.backward()
            optimizer.step()
            ep_loss += cost.item()

        scheduler.step()
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"  Fold {fold} Epoch {epoch + 1} Loss: {ep_loss / len(loader):.4f}")

    save_dir = os.path.join(config["result_root"], f"fold{fold}")
    ckpt_path = os.path.join(save_dir, "model_final.pth")

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(net.module.state_dict(), ckpt_path)

    dist.barrier()
    return ckpt_path, save_dir


# ============================================================
                      
# ============================================================
def process_outputs(video_scores, actionness, cas, video_props, start_f, t_origin, win_size, mode, config, cola_cfg):
            
    cas_p = torch.sigmoid(cas[0])  # [T, C]
    aness_p = torch.sigmoid(actionness[0])  # [T]
    vid_p = torch.sigmoid(video_scores[0]).cpu().numpy()  # [C]

                   
    if mode == "test_full":
                            
        cas_max_val, _ = torch.max(cas_p, dim=0)
        pred_cats = np.where(cas_max_val.cpu().numpy() >= cola_cfg.CLASS_THRESH)[0]
    else:
                                          
        pred_cats = np.where(vid_p >= cola_cfg.CLASS_THRESH)[0]

    if len(pred_cats) == 0: pred_cats = np.array([np.argmax(vid_p)])

             
    cas_supp = cas_p * aness_p.unsqueeze(1)

                 
    cas_np = np.expand_dims(cas_supp.cpu().numpy(), axis=2)
    aness_np = np.tile(aness_p.cpu().numpy().reshape(-1, 1, 1), (1, cola_cfg.NUM_CLASSES, 1))

                                           
    T_current = cas_p.shape[0]
    prop_dict = utils.get_proposal_dict(cas_np, aness_np, pred_cats, vid_p, T_current, cola_cfg)

             
    for cls_id, props in prop_dict.items():
        for p in props:
                                                   
                                                                   
            if mode == "test_full":
                scale = t_origin / T_current
                global_start = p[2] * scale
                global_end = p[3] * scale
            else:
                global_start = start_f + p[2]
                global_end = start_f + p[3]

            video_props.append([p[0], p[1], global_start, global_end])


@torch.no_grad()
def eval_fold_dual_mode_rank0(config, fold, ckpt_path, save_dir, device):
    print(f"\n>>> [Eval] Inferencing Fold {fold}...")
    cola_cfg = map_config_to_cola_cfg(config, fold)

              
    net = CoLA(cola_cfg).to(device)

                         
    print(f"   -> Loading weights from {ckpt_path}...")
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
                                            
    msg = net.load_state_dict(new_state_dict, strict=False)
    if len(msg.missing_keys) > 0:
        print(f"   ⚠️ Warning: Missing keys: {msg.missing_keys[:3]}...")

    net.eval()

                                   
    full_wrapper = CoLA_FullWrapper(net, win_len=cola_cfg.NUM_SEGMENTS, stride=cola_cfg.NUM_SEGMENTS // 2)

              
    test_ds = WeaklyOpportunityDataset(
        dataset_dir=config["dataset_dir"], loso_json=f"loso_sbj_{fold}.json", mode="test_window",
        fps=config["fps"], num_sensors=config["in_channels"], clip_sec=config["clip_sec"], normalize=True
    )

    id2name = {v: k for k, v in cola_cfg.CLASS_DICT.items()}

              
    gt_path = os.path.join(save_dir, "gt_for_anet.json")
    if not os.path.exists(gt_path):
        print("   Generating GT...")
        build_gt_for_anet(os.path.join(config["dataset_dir"], "annotations", f"loso_sbj_{fold}.json"), gt_path)

                
    for mode in ["test_window", "test_full"]:
        final_res = {'version': 'VERSION 1.3', 'results': {}, 'external_data': {}}
        win_size = cola_cfg.NUM_SEGMENTS
        stride = win_size // 2
        inf_time_list = []
        gpu_mem_list = []
               
        if mode == "test_window":
            iterator = get_inference_data(test_ds, mode, win_size, stride)
        else:
            def full_gen():
                for sbj in test_ds.subjects:
                    raw = test_ds._get_raw(sbj)
                    if test_ds.normalize: raw = (raw - test_ds.mean) / (test_ds.std + 1e-6)
                    yield sbj, torch.from_numpy(raw).float().unsqueeze(0), raw.shape[0]

            iterator = full_gen()

        for sbj, data_source, t_origin in tqdm(iterator, desc=mode, leave=False):
            video_props = []

            if mode == "test_full":
                chunk = data_source.to(device)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                start_t = time.time()
                video_scores, actionness, cas = full_wrapper(chunk)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_t = time.time()

                         
                         
                inf_time_list.append((end_t - start_t) * 1000.0)

                         
                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                    gpu_mem_list.append(peak_mem)
                process_outputs(video_scores, actionness, cas, video_props, 0, t_origin, 0, mode, config, cola_cfg)
            else:
                for chunk, start_f in data_source:
                    video_scores, _, actionness, cas = net(chunk.to(device))
                    process_outputs(video_scores, actionness, cas, video_props, start_f, t_origin, win_size, mode,
                                    config, cola_cfg)

            # Global NMS & Convert to Seconds
            sbj_list = []
            if video_props:
                cls_groups = {}
                for p in video_props: (cls_groups.setdefault(int(p[0]), [])).append(p)
                for cid, props in cls_groups.items():
                    keep = utils.nms(props, cola_cfg.NMS_THRESH)
                    for k in keep:
                        sbj_list.append({
                            'label': id2name.get(cid, str(cid)),
                            'score': float(k[1]),
                            'segment': [round(float(k[2]) / config["fps"], 2), round(float(k[3]) / config["fps"], 2)]
                        })
            final_res['results'][sbj] = sbj_list

               
        pred_path = os.path.join(save_dir, f"predictions_{mode}.json")
        with open(pred_path, 'w') as f:
            json.dump(final_res, f, indent=2)
        if mode == "test_full" and inf_time_list:
            avg_time = np.mean(inf_time_list)
            std_time = np.std(inf_time_list)
            avg_mem = np.mean(gpu_mem_list)
            std_mem = np.std(gpu_mem_list)

            stats_info = {
                "test_mode": mode,
                "num_samples": len(inf_time_list),
                                               
                "avg_inference_time_ms": f"{avg_time:.2f} ± {std_time:.2f}",
                "avg_gpu_memory_mb": f"{avg_mem:.2f} ± {std_mem:.2f}",
                "raw_time_list": inf_time_list,                   
                "raw_mem_list": gpu_mem_list
            }

            stats_path = os.path.join(save_dir, f"inference_stats_{mode}.json")
            with open(stats_path, 'w') as f:
                json.dump(stats_info, f, indent=2)

            print(f"   📊 [Stats] Time: {avg_time:.2f}ms | Mem: {avg_mem:.2f}MB")

        ev = ANETdetection(gt_path, pred_path, subset="test", tiou_thresholds=np.linspace(0.3, 0.7, 5),
                           verbose=False)
        _, avg, _ = ev.evaluate()
        print(f"  --> {mode}: mAP@Avg = {avg:.4f}")


# ============================================================
     
# ============================================================
def main():
    rank, local_rank, world_size = setup_ddp()

                    
    base_config = {
        "dataset_dir": "/home/lipei/TAL_data/opportunity/",
        "pretrained_dir": "/home/lipei/project/WSDDN/OtherData/Opportunity/pre_train/CNN1D",
        "result_root": "./output_opportunity_cola_ddp",

        "fps": 30,
        "in_channels": 113,
        "num_classes": 17,
        "clip_sec": 30.0,
        "clip_overlap": 0.5,

        "folds": [0, 1, 2, 3],  # 4 Fold LOSO
        "pretrained_model_name": "CNN1D",

        "training": {
            "batch_size": 16,                      
            "num_epochs": 80,
            "lr": 1e-4,
            "lr_step_size": 10,
            "lr_gamma": 0.9,
            "train_backbone": False
        },
        "cola": {
            "lambda": 0.01,
            "r_easy": 15, "r_hard": 20,
            "m": 1, "M": 20,
            "class_thresh": 0, "nms_thresh": 0.4
        }
    }

    seeds = [2022, 2024, 2026]

    for seed in seeds:
        if rank == 0: print(f"\n\n### SEED: {seed} ###")
        set_seed(seed)
        config = copy.deepcopy(base_config)
        config["seed"] = seed
        config["result_root"] = os.path.join(base_config["result_root"], f"seed_{seed}")

        for fold in config["folds"]:
            ckpt, save_dir = train_fold_ddp(config, fold, rank, local_rank, world_size)

            if rank == 0:
                eval_fold_dual_mode_rank0(config, fold, ckpt, save_dir, torch.device(f"cuda:{local_rank}"))
            dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()