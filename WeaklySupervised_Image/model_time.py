# import os
# import json
#
#
# def calculate_model_statistics(base_path, seeds, models):
#     """
#     计算每个模型的平均推理时间和GPU内存占用。
#
#     Args:
#         base_path (str): 存放所有实验结果的根目录。
#         seeds (list): 随机种子目录名称的列表。
#         models (list): 模型目录名称的列表。
#
#     Returns:
#         dict: 包含每个模型统计结果的字典。
#     """
#     results = {}
#
#     print("开始处理，请稍候...")
#     # 遍历每个模型
#     for model in models:
#         total_inf_time = 0
#         total_gpu_mem = 0
#         file_count = 0
#
#         print(f"\n正在处理模型: {model}")
#
#         # 遍历每个随机种子
#         for seed in seeds:
#             # 遍历每个fold (fold0 to fold23)
#             for fold_idx in range(24):
#                 fold_name = f"fold{fold_idx}"
#                 file_path = os.path.join(base_path, seed, model, fold_name, "inference_stats_test_full.json")
#
#                 # 检查JSON文件是否存在
#                 if os.path.exists(file_path):
#                     try:
#                         with open(file_path, 'r') as f:
#                             data = json.load(f)
#                             # 累加推理时间和GPU占用
#                             total_inf_time += data.get("avg_inf_time_ms", 0)
#                             total_gpu_mem += data.get("avg_gpu_mem_mb", 0)
#                             file_count += 1
#                     except json.JSONDecodeError:
#                         print(f"警告: 文件格式错误，无法解析: {file_path}")
#                     except KeyError:
#                         print(f"警告: 文件中缺少必要的键: {file_path}")
#                 else:
#                     print(f"警告: 文件未找到: {file_path}")
#
#         # 如果找到了文件，则计算平均值
#         if file_count > 0:
#             avg_inf_time = total_inf_time / file_count
#             avg_gpu_mem = total_gpu_mem / file_count
#             results[model] = {
#                 "average_inference_time_ms": avg_inf_time,
#                 "average_gpu_memory_mb": avg_gpu_mem,
#                 "files_processed": file_count
#             }
#             print(f"模型 {model} 处理完成，共处理 {file_count} 个文件。")
#         else:
#             results[model] = {
#                 "error": "没有找到任何有效的统计文件。"
#             }
#             print(f"模型 {model} 未找到任何文件。")
#
#     return results
#
#
# if __name__ == "__main__":
#     # --- 配置区域 ---
#     # 根目录路径
#     base_path = "/home/lipei/project/WSDDN/test_results/WEAR/"
#     # 随机种子目录列表
#     seeds = ["2022", "2024", "2026"]
#     # 模型目录列表
#     models = ["wsddn_01", "oicr_01", "pcl_01"]
#
#     # --- 执行计算 ---
#     final_results = calculate_model_statistics(base_path, seeds, models)
#
#     # --- 打印最终结果 ---
#     print("\n" + "=" * 40)
#     print("         最终统计结果")
#     print("=" * 40)
#     for model, data in final_results.items():
#         print(f"\n模型: {model}")
#         if "error" in data:
#             print(f"  错误: {data['error']}")
#         else:
#             print(f"  平均推理时间 (ms): {data['average_inference_time_ms']:.4f}")
#             print(f"  平均GPU内存占用 (MB): {data['average_gpu_memory_mb']:.4f}")
#             print(f"  成功处理的文件数量: {data['files_processed']}")
#     print("\n" + "=" * 40)
#


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. 数据准备 ---
# 将数据定义在主逻辑中，方便传递给各个绘图函数
MODEL_DATA = {
    'DCASE': 1180514,
    'CDur': 374163,
    'WSDNN': 4226620,
    'OICR': 31907948,
    'PCL': 31907948,
    'RSKP': 274944,
    'CoLA': 3992454,
}


# --- 2. 绘图函数定义 ---

# def plot_lollipop_chart(data):
#     """方案1：生成棒棒糖图"""
#     df = pd.DataFrame(list(data.items()), columns=['Model', 'Parameters'])
#     df = df.sort_values(by='Parameters', ascending=True)
#
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(12, 4))
#
#     # 创建“杆”
#     ax.hlines(y=df['Model'], xmin=0, xmax=df['Parameters'], color='skyblue', alpha=0.6, linewidth=2)
#     # 创建“头”
#     ax.scatter(df['Parameters'], df['Model'], color='dodgerblue', s=150, alpha=1, zorder=3)
#
#     # 添加数据标签
#     for index, row in df.iterrows():
#         label = f'{row["Parameters"] / 1e6:.1f}M' if row["Parameters"] >= 1e6 else f'{row["Parameters"] / 1e3:.1f}K'
#         ax.text(row['Parameters'] + 800000, row['Model'], label, color='black', ha='left', va='center')
#
#     # 美化
#     # ax.set_title('Model Parameters', fontsize=18, fontweight='bold', pad=20)
#     # ax.set_xlabel('Number of Parameters', fontsize=14)
#     ax.set_xscale('log')  # 棒棒糖图同样推荐使用对数刻度
#     ax.set_ylabel('')
#     sns.despine(left=True, bottom=True)
#     plt.tight_layout()
#     plt.show()


# def plot_donut_chart(data):
#     """方案2：生成圆环图"""
#     df = pd.DataFrame(list(data.items()), columns=['Model', 'Parameters'])
#     df = df.sort_values(by='Parameters', ascending=False)
#
#     # 筛选出参数最多的前几个，其余的归为"Others"
#     top_n = 4
#     df_top = df.iloc[:top_n]
#     others_sum = df.iloc[top_n:]['Parameters'].sum()
#     if others_sum > 0:
#         df_top = pd.concat([df_top, pd.DataFrame([{'Model': 'Others', 'Parameters': others_sum}])], ignore_index=True)
#
#     labels = df_top['Model']
#     sizes = df_top['Parameters']
#
#     # 突出最大的部分
#     explode = [0.05] + [0] * (len(labels) - 1)
#
#     plt.style.use('default')
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     # 使用柔和的颜色
#     colors = sns.color_palette('pastel')
#
#     wedges, texts, autotexts = ax.pie(
#         sizes, labels=labels, autopct='%1.1f%%',
#         startangle=90, pctdistance=0.85, explode=explode,
#         colors=colors, textprops={'fontsize': 12}
#     )
#
#     # 画一个白色的圆，变成圆环
#     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#     ax.add_artist(centre_circle)
#
#     # 美化
#     ax.axis('equal')
#     ax.set_title('Proportion of Model Parameters', fontsize=18, fontweight='bold', pad=20)
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_minimalist_barchart(data):
#     """方案3：生成信息图风格的条形图"""
#     df = pd.DataFrame(list(data.items()), columns=['Model', 'Parameters'])
#     df['Parameters_M'] = df['Parameters'] / 1_000_000
#     df = df.sort_values(by='Parameters', ascending=True)
#
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(12, 7))
#
#     # 使用统一的基础色，并突出显示最大的条
#     colors = ['lightgray'] * (len(df) - 2) + ['cornflowerblue'] * 2
#     bars = ax.barh(df['Model'], df['Parameters_M'], color=colors)
#
#     # 移除所有不必要的线
#     ax.grid(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#
#     # 移除刻度
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#     ax.set_xticks([])  # 隐藏X轴刻度
#
#     # 在条形内部/旁边添加标签
#     for bar in bars:
#         width = bar.get_width()
#         label_text = f'{width:.1f} M'
#         ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, label_text, va='center', ha='left', fontsize=11,
#                 color='black')
#
#     # Y轴标签使用text手动添加，以获得更好的对齐
#     ax.tick_params(axis='y', labelsize=14, labelcolor='black')
#
#     # 添加标题和副标题
#     fig.text(0.125, 0.95, 'OICR & PCL Dominate in Model Size', fontsize=20, fontweight='bold', ha='left')
#     fig.text(0.125, 0.90, 'Comparison of model parameters in millions (M)', fontsize=14, ha='left', color='dimgray')
#
#     plt.tight_layout(rect=[0, 0, 1, 0.9])  # 为标题留出空间
#     plt.show()


def plot_grouped_color_lollipop_chart(
        data,
        min_circle_size=60,
        max_circle_size=600,
        label_offset_points=15,
        figsize=(12, 8)
):
    """
    使用您在LaTeX中定义的精确颜色，来为模型分组着色。
    """

    # 1. 定义从LaTeX转换过来的颜色映射
    #    并根据您的最新要求分配给模型组
    color_group_map = {
        # 组1 -> 绿色
        'DCASE': '#90EE90',
        'CDur': '#90EE90',

        # 组2 -> 橙色
        'WSDNN': '#FFC069',
        'OICR': '#FFC069',
        'PCL': '#FFC069',

        # 组3 -> 红色
        'RSKP': '#FF8787',
        'CoLA': '#FF8787',
    }

    # 准备数据，按参数量从小到大排序
    df = pd.DataFrame(list(data.items()), columns=['Model', 'Parameters'])
    df = df.sort_values(by='Parameters', ascending=True)

    # 根据排序后的模型列表，从映射中查找颜色
    assigned_colors = [color_group_map.get(model, '#a9a9a9') for model in df['Model']]

    # --- 计算动态圆点大小 (与之前相同) ---
    log_params = np.log10(df['Parameters'])
    min_log, max_log = log_params.min(), log_params.max()
    if max_log == min_log:
        normalized_sizes = np.full_like(log_params, min_circle_size)
    else:
        normalized_sizes = min_circle_size + \
                           (log_params - min_log) / (max_log - min_log) * (max_circle_size - min_circle_size)

    # --- 绘图 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    # 使用新生成的、按分组定义的颜色列表
    ax.hlines(y=df['Model'], xmin=0, xmax=df['Parameters'], colors=assigned_colors, alpha=0.6, linewidth=2)
    ax.scatter(df['Parameters'], df['Model'], s=normalized_sizes, c=assigned_colors, alpha=1, zorder=3)

    # --- 添加标签和美化 (与之前相同) ---
    for index, row in df.iterrows():
        label_text = f'{row["Parameters"] / 1e6:.1f}M' if row[
                                                              "Parameters"] >= 1e6 else f'{row["Parameters"] / 1e3:.1f}K'
        ax.annotate(
            text=label_text, xy=(row['Parameters'], row['Model']),
            xytext=(label_offset_points, 0), textcoords='offset points',
            va='center', ha='left', fontsize=17
        )

    # ax.set_title('Model Parameters (Grouped by Color)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Parameters (Log Scale) ', fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xscale('log')
    ax.set_ylabel('')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


# --- 3. 主逻辑：依次调用绘图函数 ---
if __name__ == "__main__":

    plot_grouped_color_lollipop_chart(MODEL_DATA,figsize=(12, 5))

    # print("\n方案2: 生成圆环图...")
    # plot_donut_chart(MODEL_DATA)
    #
    # print("\n方案3: 生成信息图风格条形图...")
    # plot_minimalist_barchart(MODEL_DATA)

