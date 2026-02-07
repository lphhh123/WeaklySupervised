import json
import os


def transform_json():
    # 路径定义
    input_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_loso_2022/all_6_30_3/predictions_test_full.json"
    mapping_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/label_mapping.json"
    output_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_loso_2022/all_6_30_3/formatted_predictions_test_full.json"

    # 1. 加载映射文件
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
        # 注意：json中的key通常是字符串，我们需要确保匹配正确
        id_to_action = mapping_data["id_to_action"]

    # 2. 加载原始预测数据
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 3. 开始转换格式
    formatted_results = {}

    for file_name, predictions in raw_data.items():
        file_list = []
        for pred in predictions:
            # 预测条目格式: [start, end, label_id, score]
            start_time = pred[0]
            end_time = pred[1]
            label_id = str(pred[2])  # 转为字符串以匹配字典key
            score = pred[3]

            # 获取动作名称，如果不存在则显示 Unknown
            label_name = id_to_action.get(label_id, f"Unknown_{label_id}")

            # 构建目标字典
            entry = {
                "segment": [start_time, end_time],
                "label": label_name,
                "score": float(score)
            }
            file_list.append(entry)

        formatted_results[file_name] = file_list

    # 4. 包裹在 results 键下
    final_output = {
        "results": formatted_results
    }

    # 5. 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"整理完成！结果已保存至: {output_path}")


if __name__ == "__main__":
    transform_json()