import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import average_precision_score

        
sys.path.append(os.getcwd())
from core.model import Actionness_Module
from core.config_xrfv2 import cfg
from core.dataset_xrfv2 import XRFV2Dataset


def setup_ddp():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size
    else:
        print("DDP environment not detected; using single-GPU mode")
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


                       
class SimpleClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
                                                                     
        self.base_model = Actionness_Module(cfg.FEATS_DIM, cfg.NUM_CLASSES, cfg)

    def forward(self, x):
        # x: [B, 2048, 36]
        # output: embeddings, cas, actionness
        _, cas, _ = self.base_model(x)
        # cas: [B, 2048, NumClasses] (Logits before softmax)

                                       
                                   
                                             
        k = max(1, cas.shape[1] // 8)
        topk_scores, _ = torch.topk(cas, k=k, dim=1)
        video_logits = torch.mean(topk_scores, dim=1)  # [B, NumClasses]

        return video_logits


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, label, _, _, _ in loader:
            data = data.to(device)
            # forward
            logits = model(data)
            scores = torch.sigmoid(logits)

            all_preds.append(scores.cpu().numpy())
            all_targets.append(label.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

                                 
    try:
                                               
                              
        val_mAP = average_precision_score(all_targets, all_preds, average='macro')
    except:
        val_mAP = 0.0

    return val_mAP


def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

             
                    
    cfg.TRAIN_BACKBONE = True                  
    cfg.BATCH_SIZE = 32
    LR = 1e-4 * 4          
    EPOCHS = 40

    if rank == 0:
        print("Classifier-only warm-up training")
        print(f"Backbone: {cfg.BACKBONE_TYPE}")
        print(f"LR: {LR}, Epochs: {EPOCHS}")
        print(f"Save Path: {cfg.MODEL_PATH}")

        if not os.path.exists(cfg.MODEL_PATH): os.makedirs(cfg.MODEL_PATH)

             
    model = SimpleClassifier(cfg).to(device)

                
    for p in model.parameters():
        p.requires_grad = True

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

             
    train_dataset = XRFV2Dataset(mode='train', modal=cfg.MODAL, num_segments=cfg.NUM_SEGMENTS,
                                 class_dict=cfg.CLASS_DICT, seed=cfg.SEED, supervision='weak')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                                               sampler=train_sampler, num_workers=4, pin_memory=True)

                       
    if rank == 0:
        test_dataset = XRFV2Dataset(mode='test', modal=cfg.MODAL, num_segments=cfg.NUM_SEGMENTS,
                                    class_dict=cfg.CLASS_DICT, seed=cfg.SEED, supervision='weak')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

                   
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()                

    best_acc = 0.0

             
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0
        steps = 0

        for data, label, _, _, _ in train_loader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

            if rank == 0 and steps % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] Step [{steps}] Loss: {loss.item():.4f}")

            
        if rank == 0:
            val_mAP = validate(model, test_loader, device)
            print(
                f"Epoch [{epoch + 1}] Done. Avg Loss: {epoch_loss / steps:.4f} | Val Classification mAP: {val_mAP:.2%}")

            if val_mAP > best_acc:
                best_acc = val_mAP
                                                              
                                                    
                                             
                state_dict = model.module.base_model.state_dict() if world_size > 1 else model.base_model.state_dict()

                save_path = os.path.join(cfg.MODEL_PATH, "classifier_best.pth")
                torch.save(state_dict, save_path)
                print(f"Best model saved to {save_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
