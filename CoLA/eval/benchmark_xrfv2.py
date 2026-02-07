import os
import sys
import torch
import numpy as np
import json
import time
from terminaltables import AsciiTable

import core.utils as utils
from core.model import CoLA
from core.config_xrfv2 import cfg
from core.dataset_xrfv2 import XRFV2Dataset
from main_cola import test_all


def run_inference():
    print("======== 重新推理 (Re-Inference) ========")


    CORRECT_DICT = {
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
    cfg.CLASS_DICT = CORRECT_DICT
    cfg.NUM_CLASSES = len(CORRECT_DICT)


    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ckpt_path = os.path.join('output_ddp', 'cnn1d_0124_1710', 'checkpoints', 'model_best.pth')
    ckpt_dir = os.path.dirname(ckpt_path)

    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到模型权重: {ckpt_path}")
        return


    net = CoLA(cfg)
    net = net.to(device)
    state_dict = torch.load(ckpt_path)
    new_state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    net.eval()
    print(f"✅ 模型加载成功: {ckpt_path}")


    test_loader = torch.utils.data.DataLoader(
        XRFV2Dataset(mode='test', modal=cfg.MODAL, num_segments=cfg.NUM_SEGMENTS,
                     class_dict=cfg.CLASS_DICT, seed=cfg.SEED, supervision='weak'),
        batch_size=1, shuffle=False, num_workers=4
    )


    inf_times = []
    mem_peaks = []

    print("🚀 正在评估推理速度和显存使用...")
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 60: break

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start = time.time()
            _ = net(data.to(device))

            if device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = (time.time() - start) * 1000

            if i >= 10:
                inf_times.append(elapsed)
                if device.type == 'cuda':
                    mem_peaks.append(torch.cuda.max_memory_allocated() / 1024 / 1024)


    perf_results = {
        "avg_inference_time_ms": round(float(np.mean(inf_times)), 4),
        "std_inference_time_ms": round(float(np.std(inf_times)), 4),
        "peak_gpu_memory_mb": round(float(np.max(mem_peaks)), 4) if mem_peaks else 0.0
    }


    perf_file_path = os.path.join(ckpt_dir, "inference_perf_benchmark.json")
    with open(perf_file_path, "w") as f:
        json.dump(perf_results, f, indent=4)

    print(f"✅ 性能数据已保存至: {perf_file_path}")
    print(f"   [速度: {perf_results['avg_inference_time_ms']} ms | 显存: {perf_results['peak_gpu_memory_mb']} MB]")


    print("\n继续运行标准评估流程...")
    test_info = {"step": [], "test_acc": [], "average_mAP": []}
    for i in cfg.TIOU_THRESH:
        test_info[f"mAP@{i:.1f}"] = []

    cfg.OUTUT_PATH = 'output_ddp'
    test_all(net, cfg, test_loader, test_info, 0, None)
    print(f"\n======== 评估完成 ========")
    print(utils.table_format(test_info, cfg.TIOU_THRESH, '[CoLA] XRFV2 Re-Inference'))


if __name__ == '__main__':
    run_inference()