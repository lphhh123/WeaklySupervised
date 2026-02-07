import os
import json
import numpy as np
import subprocess
from terminaltables import AsciiTable

# ================= 配置区域 =================
# 1. 结果根目录 (由于脚本在 eval 文件夹，结果在上一级目录)
RESULTS_ROOT = "./output_opportunity_cola_ddp"
# 2. 种子列表
SEEDS = [2022, 2024, 2026]
# 3. 评估脚本路径 (同级目录)
EVAL_SCRIPT = "eval/eval_folds_metrics.py"
# 4. tIoU 阈值列表 (0.3-0.7)
TIOUS = [0.3, 0.4, 0.5, 0.6, 0.7]

# ===========================================

def get_metrics_structure():
    return {
        "mAP_mean": [],
        "mAP_per_tiou": [],  # [[0.3, 0.4, 0.5, 0.6, 0.7], ...]

        # 包含 Null (背景类) 的 P/R/F1
        "P_macro": [],
        "R_macro": [],
        "F1_macro": [],

        # 不含 Null 的 P/R/F1
        "P_macro_nonnull": [],
        "R_macro_nonnull": [],
        "F1_macro_nonnull": [],

        # UODIFM 6项错误率
        "UR": [], "OR": [], "DR": [], "IR": [], "FR": [], "MR": []
    }


def run_seed_eval(seed_path):
    """调用学姐脚本计算该种子下 4 Fold 的平均值"""
    # 如果路径包含空格或特殊字符，建议使用 list 形式传参更安全
    cmd = [
        "python", EVAL_SCRIPT,
        "--exp_dirs", seed_path,
        "--tiou", ",".join(map(str, TIOUS)),
        "--use_conf_thresh", "--conf_thresh_override", "0"
    ]
    # 执行并等待完成，静默输出
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    # 模式通常包含 test_full 和 test_window
    modes = ["test_full", "test_window"]
    final_stats = {m: get_metrics_structure() for m in modes}

    print(f"📍 正在读取结果根目录: {os.path.abspath(RESULTS_ROOT)}")

    for seed in SEEDS:
        seed_dir = os.path.abspath(os.path.join(RESULTS_ROOT, f"seed_{seed}"))
        if not os.path.exists(seed_dir):
            print(f"❌ 找不到种子目录: {seed_dir}")
            continue

        print(f"🚀 处理 Seed {seed}: 正在执行 4-Fold LOSO 汇总评估...")

        # 1. 运行评估脚本生成 metrics_summary_all_folds.json
        try:
            run_seed_eval(seed_dir)
        except Exception as e:
            print(f"   ⚠️ Seed {seed} 评估脚本执行异常: {e}")
            continue

        # 2. 读取生成的汇总 JSON
        summary_path = os.path.join(seed_dir, "metrics_summary_all_folds.json")
        if not os.path.exists(summary_path):
            print(f"   ⚠️ 未能在目录中找到汇总文件: {summary_path}")
            continue

        with open(summary_path, 'r') as f:
            data = json.load(f)

            for mode in modes:
                if mode not in data['modes'] or not data['modes'][mode]:
                    continue

                # 提取平均值块 (mean_over_folds 是学姐脚本自动算好的 4 Fold 均值)
                res = data['modes'][mode]['mean_over_folds']
                acc = final_stats[mode]

                # mAP
                acc["mAP_mean"].append(res["mAP_mean"])
                acc["mAP_per_tiou"].append(res["mAP_per_tiou"])

                # P/R/F1 (Incl. Null)
                acc["P_macro"].append(res["P_macro"])
                acc["R_macro"].append(res["R_macro"])
                acc["F1_macro"].append(res["F1_macro"])

                # P/R/F1 (Excl. Null)
                acc["P_macro_nonnull"].append(res["P_macro_nonnull"])
                acc["R_macro_nonnull"].append(res["R_macro_nonnull"])
                acc["F1_macro_nonnull"].append(res["F1_macro_nonnull"])

                # UODIFM
                uod = res["UODIFM"]
                for k in ["UR", "OR", "DR", "IR", "FR", "MR"]:
                    acc[k].append(uod[k])

    # 3. 打印最终报表 (Mean ± Std of 3 Seeds)
    for mode in modes:
        print_final_table(mode, final_stats[mode])


def print_final_table(mode_name, acc):
    if not acc["mAP_mean"]: return

    print(f"\n{'=' * 35} OPPORTUNITY {mode_name.upper()} FINAL SUMMARY {'=' * 35}")
    print(f"{'Mean of 3 Seeds (4 Folds each)':^80}")

    table_data = [["Metric", "Value (Mean ± Std)"]]

    # --- 1. mAP 详情 ---
    per_tiou = np.array(acc["mAP_per_tiou"])  # [Seeds, 5]
    m_tiou = np.mean(per_tiou, axis=0)
    s_tiou = np.std(per_tiou, axis=0)

    for i, t in enumerate(TIOUS):
        table_data.append([f"mAP@{t:.1f}", f"{m_tiou[i]:.4f} ± {s_tiou[i]:.4f}"])

    table_data.append(["mAP_avg (0.3-0.7)", f"{np.mean(acc['mAP_mean']):.4f} ± {np.std(acc['mAP_mean']):.4f}"])
    table_data.append(["-" * 30, "-" * 20])

    # --- 2. P/R/F1 (包含背景类) ---
    table_data.append(["[Sample-level] Incl. Null (Background)", ""])
    for k in ["P_macro", "R_macro", "F1_macro"]:
        table_data.append([f"  {k}", f"{np.mean(acc[k]):.4f} ± {np.std(acc[k]):.4f}"])

    # --- 3. P/R/F1 (不含背景类) ---
    table_data.append(["[Sample-level] Excl. Null (Action Only)", ""])
    for k in ["P_macro_nonnull", "R_macro_nonnull", "F1_macro_nonnull"]:
        table_data.append([f"  {k}", f"{np.mean(acc[k]):.4f} ± {np.std(acc[k]):.4f}"])

    table_data.append(["-" * 30, "-" * 20])

    # --- 4. UODIFM 错误分析 ---
    table_data.append(["[Error Analysis] UODIFM", ""])
    for k in ["UR", "OR", "DR", "IR", "FR", "MR"]:
        table_data.append([f"  {k}", f"{np.mean(acc[k]):.4f} ± {np.std(acc[k]):.4f}"])

    table = AsciiTable(table_data)
    table.justify_columns[0] = 'left'
    print(table.table)


if __name__ == "__main__":
    main()