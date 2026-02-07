import os
import sys
import json
import numpy as np

        
sys.path.append(os.getcwd())
from WSDDN.tool import ANETdetection
from WSDDN.utils import build_gt_for_anet

                                          
                                         
TARGET_DIR = "./output_hangtime_cola_ddp/seed_2024/fold5"

                        
RAW_ANNO_DIR = "/home/lipei/TAL_data/hangtime/annotations"
FOLD_ID = 10                      

            
PRED_FILES = ["predictions_test_window.json", "predictions_test_full.json"]


# ===========================================

def check_and_evaluate():
    print(f"🔍 开始诊断评估环境: {TARGET_DIR}")

    # ----------------------------------------
                    
    # ----------------------------------------
    gt_path = os.path.join(TARGET_DIR, "gt_for_anet.json")
    print(f"\n[Step 1] 检查真值文件 (GT): {gt_path}")

    if not os.path.exists(gt_path):
        print("   ⚠️ GT 文件缺失，正在尝试重新生成...")
        raw_loso_path = os.path.join(RAW_ANNO_DIR, f"loso_sbj_{FOLD_ID}.json")
        if os.path.exists(raw_loso_path):
            build_gt_for_anet(raw_loso_path, gt_path)
            print("   ✅ GT 生成成功！")
        else:
            print(f"   ❌ 无法生成 GT，找不到原始标注: {raw_loso_path}")
            return
    else:
        print("   ✅ GT 文件已存在。")

                    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    gt_ids = list(gt_data['database'].keys())
    print(f"   👉 GT ID 示例 (前3个): {gt_ids[:3]}")

    # ----------------------------------------
                  
    # ----------------------------------------
    for pred_name in PRED_FILES:
        pred_path = os.path.join(TARGET_DIR, pred_name)
        print(f"\n[Step 2] 检查预测文件: {pred_name}")

        if not os.path.exists(pred_path):
            print(f"   ❌ 文件不存在: {pred_path}")
            continue

                   
        try:
            with open(pred_path, 'r') as f:
                pred_data = json.load(f)
        except json.JSONDecodeError:
            print("   ❌ JSON 格式错误 (文件可能损坏或未写完)")
            continue

        if 'results' not in pred_data:
            print("   ❌ JSON 结构错误: 缺少 'results' 键")
            continue

        pred_ids = list(pred_data['results'].keys())
        total_preds = sum([len(v) for v in pred_data['results'].values()])
        print(f"   👉 Pred ID 示例 (前3个): {pred_ids[:3]}")
        print(f"   📊 包含视频数: {len(pred_ids)}, 预测片段总数: {total_preds}")

        if total_preds == 0:
            print("   ❌ 警告: 预测结果为空 (results 字典里没有片段)！mAP 必为 0。")
            continue

                            
        common_ids = set(gt_ids) & set(pred_ids)
        if len(common_ids) == 0:
            print("   ❌ 致命错误: GT 和 Pred 的视频 ID 完全不匹配！")
            print(f"      GT:   {gt_ids[0]} (示例)")
            print(f"      Pred: {pred_ids[0]} (示例)")
            print("      请检查 dataset 代码中的 file_name 处理逻辑 (是否带 .h5 后缀?)")
            continue
        else:
            print(f"   ✅ ID 匹配成功 (共有 {len(common_ids)} 个匹配视频)")

                 
        print(f"   🚀 调用 ANETdetection 评估 {pred_name} ...")
        try:
            tious = np.linspace(0.3, 0.7, 5)
            evaluator = ANETdetection(
                ground_truth_filename=gt_path,
                prediction_filename=pred_path,
                subset="test",
                tiou_thresholds=tious,
                verbose=False
            )
            mAPs, avg_mAP, _ = evaluator.evaluate()

            print(f"\n   🌟 {pred_name} 结果:")
            print(f"   mAP@Avg: {avg_mAP:.4f}")
            print(f"   mAP@0.5: {mAPs[2]:.4f}")
        except Exception as e:
            print(f"   ❌ 评估函数崩溃: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    check_and_evaluate()