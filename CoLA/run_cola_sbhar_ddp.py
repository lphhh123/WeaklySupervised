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
from WSDDN.SBHAR.dataset_sbhar_ws import WeaklySBHARDataset
from WSDDN.utils import set_seed, build_gt_for_anet, GlobalBackboneWrapper
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
# CoLA Full Wrapper
# ============================================================
class CoLA_FullWrapper(nn.Module):
    def __init__(self, cola_model):
        super().__init__()
                   
        self.backbone = cola_model.actionness_module.backbone
        self.adapter = cola_model.actionness_module.adapter
        self.f_cls = cola_model.actionness_module.f_cls
        self.dropout = cola_model.actionness_module.dropout
        self.r_easy = cola_model.r_easy
        self.get_video_cls_scores = cola_model.get_video_cls_scores

                 
        self.win_len = 512
        self.stride = 256

                                           
                                             
        from WSDDN.utils import GlobalBackboneWrapper
        self.global_extractor = GlobalBackboneWrapper(
            self.backbone,
            win_len=self.win_len,
            seg_stride=self.stride
        )

    @torch.no_grad()
    def forward(self, x):
        """
        x: [1, T_total, C]
        """
        x = x.permute(0, 2, 1)  # [1, 21, T_total]

                                       
                                   
        feat_512, info = self.global_extractor(x, return_info=True)

                                                
        # feat_512: [1, 512, T_feat] -> feat_2048: [1, 2048, T_feat]
        feat_2048 = self.adapter(feat_512)

                  
        out = self.dropout(feat_2048)
        out = self.f_cls(out)  # [1, Classes, T_feat]

        cas = out.permute(0, 2, 1)  # [1, T_feat, Classes]
        actionness = cas.sum(dim=2)  # [1, T_feat]

               
        k_easy = max(1, cas.shape[1] // 2)
        video_scores = self.get_video_cls_scores(cas, k_easy)

        return video_scores, actionness, cas, info



# ============================================================
      
# ============================================================
def map_config_to_cola_cfg(user_config, fold):
    c = edict()
    c.DATASET_NAME = 'WEAR'
    c.MODAL = 'imu'
    c.FEATS_FPS = user_config['fps']  # 50
    c.FEATS_DIM = user_config['in_channels']  # 12
    c.NUM_CLASSES = user_config['num_classes']  # 18

    c.NUM_SEGMENTS = int(user_config['clip_sec'] * c.FEATS_FPS)
    c.UP_SCALE = 1

             
    ckpt_name = f"sbhar_{user_config['pretrained_model_name']}_pretrained_loso_sbj_{fold}.pth"
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
        "walking": 0,
        "walking_upstairs": 1,
        "walking_downstairs": 2,
        "sitting": 3,
        "standing": 4,
        "lying": 5,
        "stand-to-sit": 6,
        "sit-to-stand": 7,
        "sit-to-lie": 8,
        "lie-to-sit": 9,
        "stand-to-lie": 10,
        "lie-to-stand": 11
    }
    return c


# ============================================================
                               
# ============================================================
def load_checkpoint_robust(net, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return net
    print(f"   -> Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace('module.', '')
        name = name.replace('actionness_backbone', 'actionness_module.backbone')
        name = name.replace('actionness_adapter', 'actionness_module.adapter')
        name = name.replace('actionness_f_cls', 'actionness_module.f_cls')
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=False)
    return net


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
                    offsets = [0]; raw = np.pad(raw, ((0, win_size - t_origin), (0, 0)), mode='constant')

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
    if rank == 0: print(f"\n>>> [Train DDP] WEAR Fold {fold}...")

    cola_cfg = map_config_to_cola_cfg(config, fold)
    net = CoLA(cola_cfg).to(local_rank)

    if not cola_cfg.TRAIN_BACKBONE:
        for p in net.actionness_module.backbone.parameters(): p.requires_grad = False
        net.actionness_module.backbone.eval()

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    loso_json = f"loso_sbj_{fold}.json"
    train_ds = WeaklySBHARDataset(
        dataset_dir=config["dataset_dir"], loso_json=loso_json, mode="train",
        fps=config["fps"], num_sensors=config["in_channels"],
        clip_sec=config["clip_sec"],
        clip_overlap=config["clip_overlap"],
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
            video_scores, contrast_pairs, _, cas = net(data)
            video_scores = torch.clamp(torch.sigmoid(video_scores), 1e-6, 1 - 1e-6)

            cost, loss_dict = criterion(video_scores, label, contrast_pairs)
            sparsity_weight = 0

                                         
            cas_prob = torch.sigmoid(cas)

                                      
                                                    
            sparsity_loss = torch.mean(cas_prob)

                        
            cost = cost + sparsity_weight * sparsity_loss
            cost.backward()
            optimizer.step()
            ep_loss += cost.item()

        scheduler.step()
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"  Fold {fold} Epoch {epoch + 1} Loss: {ep_loss / len(loader):.4f}")

    save_dir = os.path.join(config["result_root"], f"seed_{config['seed']}", f"fold{fold}")
    ckpt_path = os.path.join(save_dir, "model_final.pth")

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(net.module.state_dict(), ckpt_path)

    dist.barrier()
    return ckpt_path, save_dir


# ============================================================
                      
# ============================================================
def process_outputs(video_scores, actionness, cas, video_props, start_f, win_size, mode, cola_cfg, external_scale=None):
    cas_raw = torch.sigmoid(cas[0])  # [T_feat, C]
    aness_raw = torch.sigmoid(actionness[0]).unsqueeze(1)  # [T_feat, 1]
    vid_p = torch.sigmoid(video_scores[0]).cpu().numpy()

                            
                                               
                           
    smooth_kernel = 30

    combined = torch.cat([cas_raw, aness_raw], dim=1)  # [T_feat, C+1]
    combined_t = combined.permute(1, 0).unsqueeze(0)

    combined_smooth = F.avg_pool1d(combined_t, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel // 2)
    combined_p = combined_smooth.squeeze(0).permute(1, 0)

          
    if combined_p.shape[0] != cas_raw.shape[0]:
        combined_p = combined_p[:cas_raw.shape[0], :]

    cas_p = combined_p[:, :-1]
    aness_p = combined_p[:, -1]
    hard_thresh = cola_cfg['CLASS_THRESH']
                    
    if mode == "test_full":
        cas_max_val, _ = torch.max(cas_p, dim=0)
        pred_cats = np.where(cas_max_val.cpu().numpy() >= hard_thresh)[0]
    else:
        pred_cats = np.where(vid_p >= hard_thresh)[0]

    if len(pred_cats) == 0: pred_cats = np.array([np.argmax(vid_p)])

                 
    cas_supp = cas_p * aness_p.unsqueeze(1)
    cas_np = np.expand_dims(cas_supp.cpu().numpy(), axis=2)
    num_classes = cola_cfg['NUM_CLASSES']
    aness_np = np.tile(aness_p.cpu().numpy().reshape(-1, 1, 1), (1, num_classes, 1))

                             
    T_feat = cas_p.shape[0]
    prop_dict = utils.get_proposal_dict(cas_np, aness_np, pred_cats, vid_p, T_feat, cola_cfg)

                    
                                                                   
    # test_window:
                                                      
                      

                
    if mode == "test_full":
        if external_scale is None:
            raise ValueError("test_full mode requires 'external_scale' (bin_frames)")
        scale = external_scale
    else:
                                                
        scale = win_size / T_feat

    for cls_id, props in prop_dict.items():
        for p in props:
                                    
            global_start = start_f + p[2] * scale
            global_end = start_f + p[3] * scale
            video_props.append([p[0], p[1], global_start, global_end])


@torch.no_grad()
def eval_fold_dual_mode_rank0(config, fold, ckpt_path, save_dir, device):
    print(f"\n>>> [Eval] Inferencing Fold {fold}...")
    cola_cfg = map_config_to_cola_cfg(config, fold)

    net = CoLA(cola_cfg).to(device)
    net = load_checkpoint_robust(net, ckpt_path, device)
    net.eval()
    full_wrapper = CoLA_FullWrapper(net)

    test_ds = WeaklySBHARDataset(
        dataset_dir=config["dataset_dir"], loso_json=f"loso_sbj_{fold}.json", mode="test_window",
        fps=config["fps"], num_sensors=config["in_channels"], clip_sec=500,           
        normalize=True
    )

    id2name = {v: k for k, v in cola_cfg.CLASS_DICT.items()}

             
    gt_path = os.path.join(save_dir, "gt_for_anet.json")
    if not os.path.exists(gt_path):
        build_gt_for_anet(os.path.join(config["dataset_dir"], "annotations", f"loso_sbj_{fold}.json"), gt_path)

                   
    min_sec = config["testing"].get("min_sec", 0.0)
    max_sec = config["testing"].get("max_sec", 9999.0)
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
                video_scores, actionness, cas, info = full_wrapper(chunk)
                              
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                end_t = time.time()
                inf_time_list.append((end_t - start_t) * 1000.0)

                         
                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                    gpu_mem_list.append(peak_mem)
                                             
                bin_frames = info['bin_frames']

                                        
                process_outputs(
                    video_scores, actionness, cas, video_props,
                    0, 0,                          
                    mode, cola_cfg,
                    bin_frames          
                )
            else:
                for chunk, start_f in data_source:
                    video_scores, _, actionness, cas = net(chunk.to(device))
                    process_outputs(video_scores, actionness, cas, video_props, start_f, win_size, mode,
                                    cola_cfg)

            sbj_list = []
            if video_props:
                cls_groups = {}
                for p in video_props: (cls_groups.setdefault(int(p[0]), [])).append(p)
                for cid, props in cls_groups.items():
                    keep = utils.nms(props, cola_cfg.NMS_THRESH)
                    for k in keep:
                        start_sec = float(k[2]) / config["fps"]
                        end_sec = float(k[3]) / config["fps"]
                        duration = end_sec - start_sec

                        if duration < min_sec or duration > max_sec:
                            continue
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
# Main
# ============================================================
def main():
    rank, local_rank, world_size = setup_ddp()

    base_config = {
        "dataset_dir": "/home/lipei/TAL_data/sbhar/",
        "pretrained_dir": "/home/lipei/project/WSDDN/OtherData/SBHAR/pre_train/CNN1D",
        "result_root": "./output_sbhar_cola_ddp",

        "fps": 50,
        "in_channels": 3,
        "num_classes": 12,
        "clip_sec": 60,
        "clip_overlap": 0.5,

        "folds": list(range(30)),
        "pretrained_model_name": "CNN1D",

        "training": {
            "batch_size": 16,
            "num_epochs": 80,
            "lr": 1e-4,
            "lr_step_size": 10,
            "lr_gamma": 0.9,
            "train_backbone": False
        },
        "testing": {
            "min_sec": 1,
            "max_sec": 42,
        },
        "cola": {
                                         
            "lambda": 0.01,
            "r_easy": 45,
            "r_hard": 20,
            "m": 1,
            "M": 4,
            "class_thresh": 0,
            "nms_thresh": 0.4
        }
    }

    seeds = [2022]

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