import json
import os
import numpy as np

def calculate_average_metrics():
    base_dir = "/home/yinjiaxi/wstal/WeaklySupervised-master/metric_result/"

    file_names = [
        "wear_cdur_10_60s_2022.json",
        "wear_cdur_10_60s_2024.json",
        "wear_cdur_10_60s_2026.json"
    ]

    all_results = []

    for file_name in file_names:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: file {file_name} does not exist, skipped.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_results.append(data['results'])

    if not all_results:
        print("No valid data found.")
        return

    num_files = len(all_results)

    w_maps = np.array([res['window_mode']['mAPs'] for res in all_results])
    w_avg = np.array([res['window_mode']['avg_mAP'] for res in all_results])

    f_maps = np.array([res['full_mode']['mAPs'] for res in all_results])
    f_avg = np.array([res['full_mode']['avg_mAP'] for res in all_results])

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

    output_path = os.path.join(base_dir, "wear_cdur_avg.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(avg_data, f, indent=4, ensure_ascii=False)

    print(f"Computation complete. Average results saved to: {output_path}")


if __name__ == "__main__":
    calculate_average_metrics()
