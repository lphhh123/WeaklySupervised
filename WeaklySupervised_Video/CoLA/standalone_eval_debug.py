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
    print(f"Starting evaluation environment diagnosis: {TARGET_DIR}")

    # ----------------------------------------
                    
    # ----------------------------------------
    gt_path = os.path.join(TARGET_DIR, "gt_for_anet.json")
    print(f"\n[Step 1] Check ground truth file (GT): {gt_path}")

    if not os.path.exists(gt_path):
        print("   Warning: GT file missing; attempting to regenerate...")
        raw_loso_path = os.path.join(RAW_ANNO_DIR, f"loso_sbj_{FOLD_ID}.json")
        if os.path.exists(raw_loso_path):
            build_gt_for_anet(raw_loso_path, gt_path)
            print("   GT generated successfully.")
        else:
            print(f"   Unable to generate GT; raw annotations not found: {raw_loso_path}")
            return
    else:
        print("   GT file already exists.")

                    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    gt_ids = list(gt_data['database'].keys())
    print(f"   GT ID examples (first 3): {gt_ids[:3]}")

    # ----------------------------------------
                  
    # ----------------------------------------
    for pred_name in PRED_FILES:
        pred_path = os.path.join(TARGET_DIR, pred_name)
        print(f"\n[Step 2] Check prediction file: {pred_name}")

        if not os.path.exists(pred_path):
            print(f"   File not found: {pred_path}")
            continue

                   
        try:
            with open(pred_path, 'r') as f:
                pred_data = json.load(f)
        except json.JSONDecodeError:
            print("   JSON decode error (file may be corrupted or incomplete)")
            continue

        if 'results' not in pred_data:
            print("   JSON structure error: missing 'results' key")
            continue

        pred_ids = list(pred_data['results'].keys())
        total_preds = sum([len(v) for v in pred_data['results'].values()])
        print(f"   Pred ID examples (first 3): {pred_ids[:3]}")
        print(f"   Video count: {len(pred_ids)}, total predicted segments: {total_preds}")

        if total_preds == 0:
            print("   Warning: predictions are empty (no segments in results). mAP will be 0.")
            continue

                            
        common_ids = set(gt_ids) & set(pred_ids)
        if len(common_ids) == 0:
            print("   Fatal error: GT and Pred video IDs do not match.")
            print(f"      GT:   {gt_ids[0]} (example)")
            print(f"      Pred: {pred_ids[0]} (example)")
            print("      Check file_name handling in dataset code (is .h5 suffix included?)")
            continue
        else:
            print(f"   ID match success (matched videos: {len(common_ids)})")

                 
        print(f"   Running ANETdetection for {pred_name} ...")
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

            print(f"\n   {pred_name} results:")
            print(f"   mAP@Avg: {avg_mAP:.4f}")
            print(f"   mAP@0.5: {mAPs[2]:.4f}")
        except Exception as e:
            print(f"   Evaluation failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    check_and_evaluate()
