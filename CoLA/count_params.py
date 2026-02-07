import os
import torch
import glob
import numpy as np
from terminaltables import AsciiTable

# ================= 配置区域 =================
# 项目根目录下的输出文件夹路径
PATHS = {
    # 6个标准结构数据集 (Standard LOSO Structure)
    "Opportunity": "./output_opportunity_cola_ddp",
    "Hangtime": "./output_hangtime_cola_ddp",
    "RWHAR": "./output_rwhar_cola_ddp",
    "SBHAR": "./output_sbhar_cola_ddp",
    "WEAR": "./output_wear_cola_ddp",
    "WETLAB": "./output_wetlab_cola_ddp",

    # 1个特殊结构数据集 (Random Split Structure)
    "XRFV2": "./output_ddp"
}

# 权重文件名匹配模式
FILENAME_PATTERNS = ["model_final.pth", "model_best.pth"]


# ===========================================

def count_parameters_in_ckpt(ckpt_path):
    """加载权重文件并计算参数量"""
    try:
        # 加载到 CPU 即可，不需要 GPU
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # 1. 提取 State Dict
        # 你的脚本中有时直接保存 state_dict，有时保存包含 'model_state_dict' 的字典
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # 假设整个字典就是 state_dict (例如 XRFV2 部分脚本)
                state_dict = checkpoint
        else:
            print(f"   ❌ 无法识别的文件格式: {ckpt_path}")
            return None

        # 2. 计算参数量
        total_params = 0
        for k, v in state_dict.items():
            # 排除 num_batches_tracked 等非参数 buffer
            if "num_batches_tracked" in k:
                continue
            if v.dim() == 0:  # 标量通常不是可训练参数
                continue
            total_params += v.numel()

        return total_params

    except Exception as e:
        print(f"   ❌ 读取失败 {ckpt_path}: {e}")
        return None


def get_files_standard(root_dir):
    """
    标准结构: root/seed_*/fold*/model_final.pth
    """
    files = []
    # 遍历所有 seed 和 fold
    for pattern in FILENAME_PATTERNS:
        # 使用 glob 递归查找
        search_path = os.path.join(root_dir, "seed_*", "fold*", pattern)
        found = glob.glob(search_path)
        files.extend(found)
    return files


def get_files_xrfv2(root_dir):
    """
    XRFV2 结构: root/seed_*/checkpoints/model_best.pth
    """
    files = []
    for pattern in FILENAME_PATTERNS:
        # XRFV2 通常在 checkpoints 子目录下
        search_path = os.path.join(root_dir, "cnn1d_*", "checkpoints", pattern)
        found = glob.glob(search_path)
        files.extend(found)
    return files


def main():
    print("🚀 开始统计模型参数量...")
    results = []
    headers = ["Dataset", "Files Found", "Avg Params (M)", "Bytes (MB)"]

    for name, root_path in PATHS.items():
        if not os.path.exists(root_path):
            results.append([name, "0", "N/A", "N/A"])
            continue

        # 1. 获取文件列表
        if name == "XRFV2":
            files = get_files_xrfv2(root_path)
        else:
            files = get_files_standard(root_path)

        if not files:
            results.append([name, "0", "N/A", "N/A"])
            continue

        # 2. 统计参数 (为了速度，每个数据集只抽样前 3 个文件计算，理论上同一数据集模型大小一致)
        # 如果你想统计所有文件取平均，去掉切片 [:3]
        sample_files = files[:3]
        counts = []

        for f in sample_files:
            c = count_parameters_in_ckpt(f)
            if c is not None:
                counts.append(c)

        if not counts:
            results.append([name, str(len(files)), "Error", "Error"])
            continue

        # 3. 计算结果
        avg_count = np.mean(counts)
        avg_million = avg_count / 1e6
        # 估算显存/硬盘占用 (float32 = 4 bytes)
        size_mb = (avg_count * 4) / (1024 * 1024)

        results.append([
            name,
            str(len(files)),
            f"{avg_million:.2f} M",
            f"{size_mb:.2f} MB"
        ])

    # 4. 打印表格
    table = AsciiTable([headers] + results)
    print("\n" + table.table)
    print("\n注: 参数量单位为百万(M), 大小按 float32 估算。")


if __name__ == "__main__":
    main()