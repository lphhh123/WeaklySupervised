import json
import os


def calculate_average_metrics():

    base_dir = "/home/yinjiaxi/wstal/WeaklySupervised-master/metric_result/dcase/"


    file_names = ["rwhar_2022.json", "rwhar_2024.json", "rwhar_2026.json"]


    file_paths = [os.path.join(base_dir, f) for f in file_names]



    accumulated_data = {
        "window": {},
        "full": {}
    }


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


                if valid_files_count == 1:
                    meta_info["dataset"] = data.get("dataset", "unknown")
                    meta_info["model_type"] = data.get("model_type", "unknown")


                results_node = data.get("results", {})


                for category in ["window", "full"]:

                    metrics_list = results_node.get(category, [])

                    for item in metrics_list:
                        t = item.get("threshold")


                        if t not in accumulated_data[category]:
                            accumulated_data[category][t] = {"count": 0}

                        accumulated_data[category][t]["count"] += 1


                        for key, value in item.items():
                            if key != "threshold" and isinstance(value, (int, float)):
                                current_sum = accumulated_data[category][t].get(key, 0.0)
                                accumulated_data[category][t][key] = current_sum + value

        except Exception as e:
            print(f"读取文件 {file_path} 时发生错误: {e}")

    if valid_files_count == 0:
        print("未找到有效文件，程序终止。")
        return



    final_results = {
        "window": [],
        "full": []
    }


    for category in ["window", "full"]:

        sorted_thresholds = sorted(accumulated_data[category].keys())

        for t in sorted_thresholds:
            data_item = accumulated_data[category][t]
            count = data_item["count"]


            avg_item = {"threshold": t}

            for key, total_value in data_item.items():
                if key != "count":

                    average_val = total_value / count
                    avg_item[key] = round(average_val, 2)

            final_results[category].append(avg_item)


    output_json = {
        "dataset": meta_info["dataset"],
        "seed": "average",
        "model_type": meta_info["model_type"],
        "results": final_results
    }


    output_path = os.path.join(base_dir, "rwhar_avg.json")


    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4)
        print(f"成功！平均值已写入: {output_path}")
        print(f"包含 categories: {list(final_results.keys())}")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")


if __name__ == "__main__":
    calculate_average_metrics()