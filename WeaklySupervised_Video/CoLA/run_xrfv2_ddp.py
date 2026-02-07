import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from terminaltables import AsciiTable
from datetime import datetime

# 引入核心模块
import core.utils as utils
from core.model import CoLA
from core.loss import TotalLoss
from core.config_xrfv2 import cfg
from core.dataset_xrfv2 import XRFV2Dataset
from main_cola import test_all


def setup_logger(output_dir):
    logger = logging.getLogger("CoLA_Train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(output_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_ddp():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def main():
    # 1. DDP 初始化
    rank, local_rank, world_size = setup_ddp()

    # 2. 实验目录与日志配置
    if rank == 0:
        exp_name = f"{cfg.BACKBONE_TYPE}_{datetime.now().strftime('%m%d_%H%M')}"
        cfg.OUTPUT_PATH = os.path.join('./output_ddp', exp_name)
        cfg.LOG_PATH = os.path.join(cfg.OUTPUT_PATH, 'tensorboard')
        cfg.MODEL_PATH = os.path.join(cfg.OUTPUT_PATH, 'checkpoints')

        os.makedirs(cfg.LOG_PATH, exist_ok=True)
        os.makedirs(cfg.MODEL_PATH, exist_ok=True)

        with open(os.path.join(cfg.OUTPUT_PATH, 'config.txt'), 'w') as f:
            f.write(str(cfg))

        logger = setup_logger(cfg.OUTPUT_PATH)
        writer = SummaryWriter(cfg.LOG_PATH)

        print(f"======== Experiment: {exp_name} ========")
        print(f"Backbone: {cfg.BACKBONE_TYPE}")
        print(f"Pretrained: {cfg.PRETRAINED_PATH}")
        print(f"Output Dir: {cfg.OUTPUT_PATH}")
        print(f"==========================================")
    else:
        logger = None
        writer = None

    # 3. [修正] 移除手动 LR 缩放，完全信任 Config
    # Config 中 cfg.BATCH_SIZE 是单卡 Batch
    # Config 中 cfg.LR_BASE 是基础学习率 (1e-4)
    # 这种方式最清晰，如果需要多卡加速，手动在 Config 里把 LR_BASE 调大即可，不要在代码里乘来乘去。

    # 4. 构建模型
    net = CoLA(cfg).to(local_rank)

    # =======================================================
    # [新增] 智能加载预训练权重 (支持 Backbone 或 Full Model)
    # =======================================================
    ckpt_path = cfg.PRETRAINED_PATH
    if ckpt_path and os.path.exists(ckpt_path):
        if rank == 0:
            print(f"🔄 Loading Weights from: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location='cpu')

        # 尝试加载为 Full Model (Stage 1 Classifier)
        # 我们的 classifier_best.pth 保存的是 base_model (即 Actionness_Module) 的 state_dict
        # 里面的 keys 包含 'backbone...', 'adapter...', 'f_cls...'

        # 这里的 net 是 CoLA，它包含 net.actionness_module
        try:
            # 1. 尝试直接加载进 actionness_module (适配 classifier_best.pth)
            msg = net.actionness_module.load_state_dict(checkpoint, strict=False)
            if rank == 0:
                print(f"   [Mode 1] Loaded into Actionness_Module. Missing: {len(msg.missing_keys)}")
        except Exception as e:
            if rank == 0:
                print(f"   [Mode 1] Failed, trying Mode 2. Error: {e}")

            # 2. 如果失败 (比如 keys 不匹配)，可能这只是一个纯 Backbone 权重
            # 这时应该由 factory 已经加载过了 (在 model.__init__ 里)，所以这里可以跳过
            # 或者在这里做更复杂的匹配逻辑
            pass
    # =======================================================

    # 冻结策略
    if not cfg.TRAIN_BACKBONE:
        if rank == 0:
            print(f"🔒 [Config] train_backbone=False: Freezing Backbone parameters...")
        for name, param in net.actionness_module.backbone.named_parameters():
            param.requires_grad = False
    else:
        if rank == 0:
            print(f"🔓 [Config] train_backbone=True: Backbone is TRAINABLE.")

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # 5. 数据加载
    train_dataset = XRFV2Dataset(
        mode='train',
        modal=cfg.MODAL,
        num_segments=cfg.NUM_SEGMENTS,
        class_dict=cfg.CLASS_DICT,
        seed=cfg.SEED,
        supervision='weak'
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        sampler=train_sampler,
        pin_memory=True
    )

    if rank == 0:
        test_loader = torch.utils.data.DataLoader(
            XRFV2Dataset(
                mode='test',
                modal=cfg.MODAL,
                num_segments=cfg.NUM_SEGMENTS,
                class_dict=cfg.CLASS_DICT,
                seed=cfg.SEED,
                supervision='weak'
            ),
            batch_size=1,
            shuffle=False, num_workers=4,
            pin_memory=True
        )

    # 6. 优化器 & 调度器
    # [修正] 缩进对齐
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())

    current_lr = cfg.LR_BASE

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=current_lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.LR_STEP_SIZE,
        gamma=cfg.LR_GAMMA
    )

    # 7. 训练循环
    criterion = TotalLoss().to(local_rank)
    step = 0
    best_mAP = -1
    test_info = {"step": [], "test_acc": [], "average_mAP": []}
    for i in cfg.TIOU_THRESH:
        test_info[f"mAP@{i:.1f}"] = []

    total_epochs = cfg.NUM_EPOCHS

    for epoch in range(total_epochs):
        train_sampler.set_epoch(epoch)

        net.train()
        if not cfg.TRAIN_BACKBONE:
            net.module.actionness_module.backbone.eval()

        for batch_idx, (data, label, _, _, _) in enumerate(train_loader):
            step += 1

            data = data.to(local_rank)
            label = label.to(local_rank)

            optimizer.zero_grad()
            video_scores, contrast_pairs, _, _ = net(data)
            cost, loss_dict = criterion(video_scores, label, contrast_pairs)

            if epoch < 5:
                cost = loss_dict['Loss/Action']

            cost.backward()
            optimizer.step()

            if rank == 0:
                if step % cfg.PRINT_FREQ == 0:
                    now_lr = optimizer.param_groups[0]['lr']
                    log_msg = f'Epoch: [{epoch + 1}/{total_epochs}] Step: {step} ' \
                              f'Loss: {cost.item():.4f} (Act: {loss_dict["Loss/Action"]:.4f}) ' \
                              f'LR: {now_lr:.6f}'

                    if logger: logger.info(log_msg)
                    print(log_msg)

                    if writer:
                        writer.add_scalar('Train/Total_Loss', cost.item(), step)
                        writer.add_scalar('Train/Action_Loss', loss_dict["Loss/Action"].item(), step)

        scheduler.step()

        if rank == 0 and (epoch % 5 == 0 or epoch > total_epochs - 5):
            if logger: logger.info(f"=> Evaluating at Epoch {epoch + 1}...")
            print(f"=> Evaluating at Epoch {epoch + 1}...")

            mAP_50, mAP_AVG = test_all(net, cfg, test_loader, test_info, step, writer)

            if logger: logger.info(f"   mAP@0.5: {mAP_50:.2%}, mAP@AVG: {mAP_AVG:.2%}")

            if mAP_AVG > best_mAP:
                best_mAP = mAP_AVG
                if logger: logger.info(f"⭐️ New Best mAP@AVG: {best_mAP:.2%}")
                print(f"⭐️ New Best mAP@AVG: {best_mAP:.2%}")
                torch.save(net.module.state_dict(), os.path.join(cfg.MODEL_PATH, "model_best.pth"))

    cleanup_ddp()


if __name__ == '__main__':
    main()