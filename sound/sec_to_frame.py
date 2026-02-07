import json
import os

def scale_segments():
    # 输入和输出路径定义
    input_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_loso_2022/all_6_30_3/predictions_test_window.json"
    output_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_loso_2022/all_6_30_3/predictions_test_window_50.json"

    # 1. 读取原始 JSON 文件
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 遍历并修改数据
    if "results" in data:
        for file_key in data["results"]:
            prediction_list = data["results"][file_key]
            for item in prediction_list:
                if "segment" in item and isinstance(item["segment"], list):
                    # 将 segment 中的 [start, end] 均乘以 50
                    item["segment"] = [val * 50 for val in item["segment"]]
    else:
        print("错误: JSON 文件中未找到 'results' 键")
        return

    # 3. 将处理后的数据写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"整理完成！处理后的文件已保存至: {output_path}")

if __name__ == "__main__":
    scale_segments()