import json
import numpy as np
import os

# 结果保存路径
RESULT_SAVE_DIR = '/home/yinjiaxi/wstal/WeaklySupervised-master/metric_result/'


def calculate_average_maps(file_path):
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # 2. 提取文件名标识 (从倒数第二个文件夹名称获取)
    # 例如 parent_dir 是 .../rwhar_dcase_10_500s_2022
    parent_dir = os.path.dirname(file_path)
    experiment_name = os.path.basename(parent_dir)

    print(f"检测到实验名称: {experiment_name}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 初始化列表来存储每个 fold 的数据
    window_mAPs = []
    window_avg_mAPs = []
    full_mAPs = []
    full_avg_mAPs = []

    # 3. 遍历数据提取 mAP (适配新数据结构)
    for entry in data:
        # --- 解析 Window Mode ---
        if 'window_mode' in entry:
            win_data = entry['window_mode']
            if 'mAPs' in win_data:
                window_mAPs.append(win_data['mAPs'])
            if 'avg_mAP' in win_data:
                window_avg_mAPs.append(win_data['avg_mAP'])

        # --- 解析 Full Mode ---
        if 'full_mode' in entry:
            full_data = entry['full_mode']
            if 'mAPs' in full_data:
                full_mAPs.append(full_data['mAPs'])
            if 'avg_mAP' in full_data:
                full_avg_mAPs.append(full_data['avg_mAP'])

    # 准备要写入 JSON 的字典
    output_json = {
        "experiment_name": experiment_name,
        "source_file": file_path,
        "results": {}
    }

    # 4. 计算平均值并存入字典

    # Window 模式计算
    if window_mAPs:
        window_mAPs_np = np.array(window_mAPs)
        mean_window_mAPs = np.mean(window_mAPs_np, axis=0)
        mean_window_avg_mAP = np.mean(window_avg_mAPs)

        output_json["results"]["window_mode"] = {
            "mAPs": mean_window_mAPs.tolist(),
            "avg_mAP": float(mean_window_avg_mAP)
        }

        print("-" * 30)
        print("Window Mode (Window 模式):")
        print(f"Thresholds mAPs: {mean_window_mAPs}")
        print(f"Average mAP: {mean_window_avg_mAP}")
    else:
        print("未找到 Window 模式数据")

    # Full 模式计算
    if full_mAPs:
        full_mAPs_np = np.array(full_mAPs)
        mean_full_mAPs = np.mean(full_mAPs_np, axis=0)
        mean_full_avg_mAP = np.mean(full_avg_mAPs)

        output_json["results"]["full_mode"] = {
            "mAPs": mean_full_mAPs.tolist(),
            "avg_mAP": float(mean_full_avg_mAP)
        }

        print("-" * 30)
        print("Full Mode (Full 模式):")
        print(f"Thresholds mAPs: {mean_full_mAPs}")
        print(f"Average mAP: {mean_full_avg_mAP}")
    else:
        print("未找到 Full 模式数据")

    print("-" * 30)

    # 5. 保存 JSON 文件
    if not os.path.exists(RESULT_SAVE_DIR):
        os.makedirs(RESULT_SAVE_DIR)
        print(f"创建目录: {RESULT_SAVE_DIR}")

    save_filename = f"{experiment_name}.json"
    save_path = os.path.join(RESULT_SAVE_DIR, save_filename)

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)
        print(f"结果已成功写入: {save_path}")
    except Exception as e:
        print(f"写入文件失败: {e}")


if __name__ == "__main__":
    # 更新后的目标文件路径
    target_file = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/rwhar_dcase_10_500s_2026/all_folds_results_500s.json"

    print(f"正在处理文件: {target_file}")
    calculate_average_maps(target_file)