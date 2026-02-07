import os
import json
import numpy as np
import subprocess
import shutil

                                          
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "output_ddp")
EVAL_SCRIPT_PATH = os.path.join(CURRENT_SCRIPT_DIR, "eval_xrfv2_metrics.py")

MODES = ["test_full", "test_window"]
TIOUS = [0.3, 0.4, 0.5, 0.6, 0.7]             


# ===========================================

def get_metrics_template():
    return {
        "mAP_mean": [],
        "mAP_per_tiou": [],
        "P_macro": [], "R_macro": [], "F1_macro": [],
        "P_macro_nonnull": [], "R_macro_nonnull": [], "F1_macro_nonnull": [],
        "UR": [], "OR": [], "DR": [], "IR": [], "FR": [], "MR": []
    }


def print_summary_table(mode_name, acc, valid_count):
    from terminaltables import AsciiTable
    if valid_count == 0:
        return

    print(f"\n{'=' * 30} {mode_name} Summary ({valid_count} Files) {'=' * 30}")

           
    table_data = [[f"Metric", "Mean", "Std"]]

                                        
    if acc['mAP_per_tiou']:
        per_tiou_matrix = np.array(acc['mAP_per_tiou'])                
        mean_per_tiou = np.mean(per_tiou_matrix, axis=0)
        std_per_tiou = np.std(per_tiou_matrix, axis=0)

        for i, t in enumerate(TIOUS):
            table_data.append([f"mAP@{t:.1f}", f"{mean_per_tiou[i]:.4f}", f"{std_per_tiou[i]:.4f}"])

                    
    if acc['mAP_mean']:
        mean_val = np.mean(acc['mAP_mean'])
        std_val = np.std(acc['mAP_mean'])
        table_data.append(["mAP_avg (0.3-0.7)", f"{mean_val:.4f}", f"{std_val:.4f}"])

    table_data.append(["-" * 20, "-" * 10, "-" * 10])       

                                  
    other_keys = [
        "P_macro_nonnull", "R_macro_nonnull", "F1_macro_nonnull",
        "P_macro", "R_macro", "F1_macro",
        "UR", "OR", "DR", "IR", "FR", "MR"
    ]

    for key in other_keys:
        if key in acc and acc[key]:
            mean_val = np.mean(acc[key])
            std_val = np.std(acc[key])
            table_data.append([key, f"{mean_val:.4f}", f"{std_val:.4f}"])

    table = AsciiTable(table_data)
    table.justify_columns = {0: 'left', 1: 'center', 2: 'center'}
    print(table.table)


def main():
    accumulators = {m: get_metrics_template() for m in MODES}
    counts = {m: 0 for m in MODES}

    for mode in MODES:
        print(f"\n>>> Scanning Folder: {mode}")
        mode_dir = os.path.join(RESULTS_ROOT, mode)
        if not os.path.exists(mode_dir): continue

        all_files = [f for f in os.listdir(mode_dir) if f.endswith('.json') and 'metrics_summary' not in f]

        for json_file in all_files:
            file_base_name = os.path.splitext(json_file)[0]
            src_file_path = os.path.join(mode_dir, json_file)

                         
            eval_tmp_dir = os.path.join(mode_dir, f"tmp_eval_{file_base_name}")
            if os.path.exists(eval_tmp_dir): shutil.rmtree(eval_tmp_dir)
            os.makedirs(eval_tmp_dir)

                                                   
            shutil.copy(src_file_path, os.path.join(eval_tmp_dir, "predictions_test_full.json"))
            shutil.copy(src_file_path, os.path.join(eval_tmp_dir, "predictions_test_window.json"))

                     
            print(f"  Evaluating: {json_file} ...")
            cmd = (
                f"python {EVAL_SCRIPT_PATH} "
                f"--pred_dir {eval_tmp_dir} "
                f"--tiou 0.3,0.4,0.5,0.6,0.7 "
                f"--use_conf_thresh --conf_thresh 0.1"
            )

            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
            except Exception as e:
                print(f"    Failed to evaluate {json_file}. Error: {e}")
                continue

                     
            summary_path = os.path.join(eval_tmp_dir, "metrics_summary_xrfv2.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                    if mode in data['modes'] and data['modes'][mode]:
                        res = data['modes'][mode]
                        acc = accumulators[mode]
                        acc['mAP_mean'].append(res['mAP_mean'])
                        acc['mAP_per_tiou'].append(res['mAP_per_tiou'])
                        acc['P_macro'].append(res['P_macro'])
                        acc['R_macro'].append(res['R_macro'])
                        acc['F1_macro'].append(res['F1_macro'])
                        acc['P_macro_nonnull'].append(res['P_macro_nonnull'])
                        acc['R_macro_nonnull'].append(res['R_macro_nonnull'])
                        acc['F1_macro_nonnull'].append(res['F1_macro_nonnull'])
                        for k in ['UR', 'OR', 'DR', 'IR', 'FR', 'MR']:
                            acc[k].append(res['UODIFM'][k])
                        counts[mode] += 1

                shutil.rmtree(eval_tmp_dir)

              
    for mode in MODES:
        print_summary_table(mode, accumulators[mode], counts[mode])


if __name__ == "__main__":
    main()
