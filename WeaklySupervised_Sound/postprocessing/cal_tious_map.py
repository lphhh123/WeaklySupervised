import json
import numpy as np
import os


RESULT_SAVE_DIR = '/home/yinjiaxi/wstal/WeaklySupervised-master/metric_result/'


def calculate_average_maps(file_path):

    if not os.path.exists(file_path):
        print(f"Message: Message {file_path}")
        return



    parent_dir = os.path.dirname(file_path)
    experiment_name = os.path.basename(parent_dir)

    print(f"Message: {experiment_name}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Message: {e}")
        return


    window_mAPs = []
    window_avg_mAPs = []
    full_mAPs = []
    full_avg_mAPs = []


    for entry in data:

        if 'window_mode' in entry:
            win_data = entry['window_mode']
            if 'mAPs' in win_data:
                window_mAPs.append(win_data['mAPs'])
            if 'avg_mAP' in win_data:
                window_avg_mAPs.append(win_data['avg_mAP'])


        if 'full_mode' in entry:
            full_data = entry['full_mode']
            if 'mAPs' in full_data:
                full_mAPs.append(full_data['mAPs'])
            if 'avg_mAP' in full_data:
                full_avg_mAPs.append(full_data['avg_mAP'])


    output_json = {
        "experiment_name": experiment_name,
        "source_file": file_path,
        "results": {}
    }




    if window_mAPs:
        window_mAPs_np = np.array(window_mAPs)
        mean_window_mAPs = np.mean(window_mAPs_np, axis=0)
        mean_window_avg_mAP = np.mean(window_avg_mAPs)

        output_json["results"]["window_mode"] = {
            "mAPs": mean_window_mAPs.tolist(),
            "avg_mAP": float(mean_window_avg_mAP)
        }

        print("-" * 30)
        print("Window Mode (Window Message):")
        print(f"Thresholds mAPs: {mean_window_mAPs}")
        print(f"Average mAP: {mean_window_avg_mAP}")
    else:
        print("Message Window Message")


    if full_mAPs:
        full_mAPs_np = np.array(full_mAPs)
        mean_full_mAPs = np.mean(full_mAPs_np, axis=0)
        mean_full_avg_mAP = np.mean(full_avg_mAPs)

        output_json["results"]["full_mode"] = {
            "mAPs": mean_full_mAPs.tolist(),
            "avg_mAP": float(mean_full_avg_mAP)
        }

        print("-" * 30)
        print("Full Mode (Full Message):")
        print(f"Thresholds mAPs: {mean_full_mAPs}")
        print(f"Average mAP: {mean_full_avg_mAP}")
    else:
        print("Message Full Message")

    print("-" * 30)


    if not os.path.exists(RESULT_SAVE_DIR):
        os.makedirs(RESULT_SAVE_DIR)
        print(f"Message: {RESULT_SAVE_DIR}")

    save_filename = f"{experiment_name}.json"
    save_path = os.path.join(RESULT_SAVE_DIR, save_filename)

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)
        print(f"Message: {save_path}")
    except Exception as e:
        print(f"Message: {e}")


if __name__ == "__main__":

    target_file = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/rwhar_dcase_10_500s_2026/all_folds_results_500s.json"

    print(f"Message: {target_file}")
    calculate_average_maps(target_file)