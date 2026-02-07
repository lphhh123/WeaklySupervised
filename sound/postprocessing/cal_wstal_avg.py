import json
import os


def calculate_average_metrics():
    # 设定文件夹路径
    base_dir = "/home/yinjiaxi/wstal/WeaklySupervised-master/metric_result/dcase/"

    # 指定需要读取的文件名 (保持不变)
    file_names = ["rwhar_2022.json", "rwhar_2024.json", "rwhar_2026.json"]

    # 构造完整路径
    file_paths = [os.path.join(base_dir, f) for f in file_names]

    # 用于存储累加数据的字典
    # 新结构: { 'window': { threshold: {stats...} }, 'full': { threshold: {stats...} } }
    accumulated_data = {
        "window": {},
        "full": {}
    }

    # 用于保存元数据 (dataset name, model_type 等，取第一个读取到的文件的值)
    meta_info = {
        "dataset": "",
        "model_type": ""
    }

    valid_files_count = 0
    print(f"开始处理路径下的文件: {base_dir}")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}，已跳过。")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                valid_files_count += 1

                # 如果是读取的第一个文件，保存一下元数据
                if valid_files_count == 1:
                    meta_info["dataset"] = data.get("dataset", "unknown")
                    meta_info["model_type"] = data.get("model_type", "unknown")

                # 获取 results 节点
                results_node = data.get("results", {})

                # 遍历两个主要类别: window 和 full
                for category in ["window", "full"]:
                    # 获取该类别下的指标列表
                    metrics_list = results_node.get(category, [])

                    for item in metrics_list:
                        t = item.get("threshold")

                        # 初始化该类别下、该阈值的数据容器
                        if t not in accumulated_data[category]:
                            accumulated_data[category][t] = {"count": 0}

                        accumulated_data[category][t]["count"] += 1

                        # 累加各个指标 (除了 threshold 本身)
                        for key, value in item.items():
                            if key != "threshold" and isinstance(value, (int, float)):
                                current_sum = accumulated_data[category][t].get(key, 0.0)
                                accumulated_data[category][t][key] = current_sum + value

        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")

    if valid_files_count == 0:
        print("未找到有效文件，程序终止。")
        return

    # --- 计算平均值并构建结果 ---

    final_results = {
        "window": [],
        "full": []
    }

    # 遍历 window 和 full
    for category in ["window", "full"]:
        # 获取该类别下的所有阈值并排序
        sorted_thresholds = sorted(accumulated_data[category].keys())

        for t in sorted_thresholds:
            data_item = accumulated_data[category][t]
            count = data_item["count"]

            # 创建该阈值的平均值对象
            avg_item = {"threshold": t}

            for key, total_value in data_item.items():
                if key != "count":
                    # 计算平均值并保留两位小数
                    average_val = total_value / count
                    avg_item[key] = round(average_val, 2)

            final_results[category].append(avg_item)

    # 构建最终输出的 JSON 结构 (适配新格式)
    output_json = {
        "dataset": meta_info["dataset"],
        "seed": "average",
        "model_type": meta_info["model_type"],
        "results": final_results
    }

    # 输出文件路径
    output_path = os.path.join(base_dir, "rwhar_avg.json")

    # 写入文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4)
        print(f"成功！平均值已写入: {output_path}")
        print(f"包含 categories: {list(final_results.keys())}")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")


if __name__ == "__main__":
    calculate_average_metrics()