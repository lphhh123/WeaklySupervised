import json
import os
import numpy as np

# 计算wstal的所有数据集在不同tious下的map的均值
def calculate_average_metrics():
    # 基础路径
    base_dir = "/home/yinjiaxi/wstal/WeaklySupervised-master/metric_result/"

    # 文件列表（根据您的描述，通常是 2022, 2023, 2024）
    file_names = [
        "wear_cdur_10_60s_2022.json",
        "wear_cdur_10_60s_2024.json",
        "wear_cdur_10_60s_2026.json"
    ]

    all_results = []

    # 1. 读取所有文件数据
    for file_name in file_names:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_name} 不存在，跳过。")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_results.append(data['results'])

    if not all_results:
        print("未找到任何有效数据。")
        return

    # 2. 初始化用于计算均值的结构
    num_files = len(all_results)

    # 提取 window_mode 数据
    w_maps = np.array([res['window_mode']['mAPs'] for res in all_results])
    w_avg = np.array([res['window_mode']['avg_mAP'] for res in all_results])

    # 提取 full_mode 数据
    f_maps = np.array([res['full_mode']['mAPs'] for res in all_results])
    f_avg = np.array([res['full_mode']['avg_mAP'] for res in all_results])

    # 3. 计算均值 (转换为 list 以便 JSON 序列化)
    avg_data = {
        "experiment_name": "wear_cdur_10_60s_average",
        "files_averaged": file_names,
        "results": {
            "window_mode": {
                "mAPs": np.mean(w_maps, axis=0).tolist(),
                "avg_mAP": float(np.mean(w_avg))
            },
            "full_mode": {
                "mAPs": np.mean(f_maps, axis=0).tolist(),
                "avg_mAP": float(np.mean(f_avg))
            }
        }
    }

    # 4. 写入结果文件
    output_path = os.path.join(base_dir, "wear_cdur_avg.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(avg_data, f, indent=4, ensure_ascii=False)

    print(f"计算完成，均值结果已保存至: {output_path}")


if __name__ == "__main__":
    calculate_average_metrics()