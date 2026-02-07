import os
import json
import glob
import numpy as np
from terminaltables import AsciiTable

# ================= 配置区域 =================
# 你的项目根目录
PROJECT_ROOT = "/home/lipei/project/CoLA"

# 需要统计的 6 个数据集名称
DATASETS = [
    "Opportunity",
    "Hangtime",
    "RWHAR",
    "SBHAR",
    "WEAR",
    "WETLAB"
]

# 目标文件名
TARGET_FILE = "inference_stats_test_full.json"


# ===========================================

def parse_value_from_str(s):
    """
    解析 "12.34 ± 1.00" 格式的字符串，提取均值部分
    """
    try:
        if "±" in s:
            return float(s.split('±')[0].strip())
        else:
            return float(s)
    except:
        return None


def get_dataset_stats(dataset_name):
    # 构造数据集输出目录: output_rwhar_cola_ddp
    dir_name = f"output_{dataset_name.lower()}_cola_ddp"
    search_root = os.path.join(PROJECT_ROOT, dir_name)

    if not os.path.exists(search_root):
        return None, None, 0

    # 递归寻找所有 inference_stats_test_full.json
    # 路径类似: output_rwhar_cola_ddp/seed_2022/fold0/inference_stats_test_full.json
    pattern = os.path.join(search_root, "**", TARGET_FILE)
    files = glob.glob(pattern, recursive=True)

    times = []
    mems = []

    for json_file in files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

                # 提取 Time
                t_str = data.get("avg_inference_time_ms", None)
                if t_str:
                    t_val = parse_value_from_str(t_str)
                    if t_val is not None: times.append(t_val)

                # 提取 Memory
                m_str = data.get("avg_gpu_memory_mb", None)
                if m_str:
                    m_val = parse_value_from_str(m_str)
                    if m_val is not None: mems.append(m_val)

        except Exception as e:
            print(f"⚠️ 读取失败: {json_file} ({e})")
            continue

    return times, mems, len(files)


def main():
    print(f"🚀 开始统计推理性能 (Target: {TARGET_FILE})")
    print(f"📂 根目录: {PROJECT_ROOT}\n")

    table_data = [["Dataset", "Files Found", "Time (ms)", "Memory (MB)"]]

    for ds in DATASETS:
        times, mems, count = get_dataset_stats(ds)

        if count == 0:
            table_data.append([ds, "0", "N/A", "N/A"])
        else:
            # 计算所有 Fold 和 Seed 的总平均值
            avg_time = np.mean(times)
            std_time = np.std(times)

            avg_mem = np.mean(mems)
            std_mem = np.std(mems)

            table_data.append([
                ds,
                str(count),
                f"{avg_time:.2f} ± {std_time:.2f}",
                f"{avg_mem:.2f} ± {std_mem:.2f}"
            ])

    # 打印表格
    table = AsciiTable(table_data)
    table.justify_columns = {0: 'left', 1: 'center', 2: 'center', 3: 'center'}
    print(table.table)
    print("\n注: Time 为单样本推理耗时, Memory 为峰值显存占用。")


if __name__ == "__main__":
    main()