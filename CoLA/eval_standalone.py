import os
import sys
import torch
import numpy as np
import json
from terminaltables import AsciiTable

# 引入核心模块
import core.utils as utils
from core.model import CoLA
from core.config_xrfv2 import cfg
from core.dataset_xrfv2 import XRFV2Dataset
from main_cola import test_all


def run_inference():
    print("======== 重新推理 (Re-Inference) ========")

    # 1. 强制覆盖 CLASS_DICT (确保万无一失)
    # 必须是: "动作名称": ID (train_label.json 里的 ID)
    CORRECT_DICT = {
        "Stretching": 0,
        "Pouring Water": 1,
        "Writing": 2,
        "Cutting Fruit": 3,
        "Eating Fruit": 4,
        "Taking Medicine": 5,
        "Drinking Water": 6,
        "Sitting Down": 7,
        "Turning On/Off Eye Protection Lamp": 8,
        "Opening/Closing Curtains": 9,
        "Opening/Closing Windows": 10,
        "Typing": 11,
        "Opening Envelope": 12,
        "Throwing Garbage": 13,
        "Picking Fruit": 14,
        "Picking Up Items": 15,
        "Answering Phone": 16,
        "Using Mouse": 17,
        "Wiping Table": 18,
        "Writing on Blackboard": 19,
        "Washing Hands": 20,
        "Using Phone": 21,
        "Reading": 22,
        "Watering Plants": 23,
        "Walking": 24,
        "Getting Out of Bed": 25,
        "Standing Up": 26,
        "Lying Down": 27,
        "Standing Still": 28,
        "Lying Still": 29
    }
    cfg.CLASS_DICT = CORRECT_DICT
    cfg.NUM_CLASSES = len(CORRECT_DICT)

    # 2. 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

    # 3. 寻找最佳模型权重
    # 假设你的权重在 output_ddp/checkpoints/model_best.pth
    ckpt_path = os.path.join('output_ddp', 'cnn1d_0124_1710', 'checkpoints', 'model_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到模型权重: {ckpt_path}")
        # 尝试找找最后的权重
        # ckpt_path = ...
        return

    print(f"Loading Model from: {ckpt_path}")

    # 4. 构建模型
    net = CoLA(cfg)
    net = net.cuda()
    # 加载权重 (注意: 如果是 DDP 训练保存的，key 可能带有 'module.')
    state_dict = torch.load(ckpt_path)

    # 修正 DDP 权重的 key
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    print("✅ 模型加载成功")

    # 5. 加载测试集
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
        shuffle=False, num_workers=4
    )

    # 6. 运行测试并生成 JSON
    print("Running Inference...")
    test_info = {"step": [], "test_acc": [], "average_mAP": []}
    for i in cfg.TIOU_THRESH:
        test_info[f"mAP@{i:.1f}"] = []

    # 调用 test_all (会自动保存 result.json 到 cfg.OUTPUT_PATH)
    # 我们临时修改 OUTPUT_PATH 以免覆盖之前的，或者直接用 output_ddp
    cfg.OUTUT_PATH = 'output_ddp'

    mAP_50, mAP_AVG = test_all(net, cfg, test_loader, test_info, 0, None)

    print(f"\n======== 重新评估结果 ========")
    print(utils.table_format(test_info, cfg.TIOU_THRESH, '[CoLA] XRFV2 Re-Inference'))
    print(f"Results saved to: {os.path.join(cfg.OUTPUT_PATH, 'result.json')}")


if __name__ == '__main__':
    run_inference()