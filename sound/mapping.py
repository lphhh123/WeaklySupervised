import json
import os


def transform_json():
          
    input_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_loso_2022/all_6_30_3/predictions_test_full.json"
    mapping_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/label_mapping.json"
    output_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_loso_2022/all_6_30_3/formatted_predictions_test_full.json"

               
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
                                       
        id_to_action = mapping_data["id_to_action"]

                 
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

               
    formatted_results = {}

    for file_name, predictions in raw_data.items():
        file_list = []
        for pred in predictions:
                                                   
            start_time = pred[0]
            end_time = pred[1]
            label_id = str(pred[2])                 
            score = pred[3]

                                     
            label_name = id_to_action.get(label_id, f"Unknown_{label_id}")

                    
            entry = {
                "segment": [start_time, end_time],
                "label": label_name,
                "score": float(score)
            }
            file_list.append(entry)

        formatted_results[file_name] = file_list

                       
    final_output = {
        "results": formatted_results
    }

             
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"整理完成！结果已保存至: {output_path}")


if __name__ == "__main__":
    transform_json()