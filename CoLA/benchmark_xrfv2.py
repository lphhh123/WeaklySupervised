import os
import sys
import torch
import numpy as np
import json
import time  # 用于计时
from terminaltables import AsciiTable

# 引入核心模块
import core.utils as utils
from core.model import CoLA
from core.config_xrfv2 import cfg
from core.dataset_xrfv2 import XRFV2Dataset
from main_cola import test_all


def run_inference():
    print("======== 重新推理 (Re-Inference) ========")

    # 1. 配置映射
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

    # 2. 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. 路径设置 (获取权重文件夹)
    ckpt_path = os.path.join('output_ddp', 'cnn1d_0123_2301', 'checkpoints', 'model_best.pth')
    ckpt_dir = os.path.dirname(ckpt_path)  # 指向 .../checkpoints/ 文件夹

    if not os.path.exists(ckpt_path):
        print(f"❌ 找不到模型权重: {ckpt_path}")
        return

    # 4. 构建并加载模型
    net = CoLA(cfg)
    net = net.to(device)
    state_dict = torch.load(ckpt_path)
    new_state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    net.eval()
    print(f"✅ 模型加载成功: {ckpt_path}")

    # 5. 加载测试集
    test_loader = torch.utils.data.DataLoader(
        XRFV2Dataset(mode='test', modal=cfg.MODAL, num_segments=cfg.NUM_SEGMENTS,
                     class_dict=cfg.CLASS_DICT, seed=cfg.SEED, supervision='weak'),
        batch_size=1, shuffle=False, num_workers=4
    )

    # 6. 推理性能专项测试 (只测前50个样本以保证稳定)
    inf_times = []
    mem_peaks = []

    print("🚀 正在评估推理速度和显存使用...")
    with torch.no_grad():
        for i, (data, *others) in enumerate(test_loader):
            if i >= 200: break  # 总测试60次，排除前10次热身
            data = data.to(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start = time.time()
            _ = net(data.to(device))

            if device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed = (time.time() - start) * 1000

            if i >= 10:  # 排除前10次热身干扰
                inf_times.append(elapsed)
                if device.type == 'cuda':
                    mem_peaks.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

    # 计算均值
    perf_results = {
        "avg_inference_time_ms": round(float(np.mean(inf_times)), 4),
        "std_inference_time_ms": round(float(np.std(inf_times)), 4),
        "peak_gpu_memory_mb": round(float(np.max(mem_peaks)), 4) if mem_peaks else 0.0
    }

    # 7. 保存性能结果到权重文件夹
    perf_file_path = os.path.join(ckpt_dir, "inference_perf_benchmark.json")
    with open(perf_file_path, "w") as f:
        json.dump(perf_results, f, indent=4)

    print(f"✅ 性能数据已保存至: {perf_file_path}")
    print(f"   [速度: {perf_results['avg_inference_time_ms']} ms | 显存: {perf_results['peak_gpu_memory_mb']} MB]")

    # 8. 运行原有的评估逻辑 (mAP等)
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