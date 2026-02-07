# import os
# import json
#
#
# def calculate_model_statistics(base_path, seeds, models):
#     """

#
#     Args:



#
#     Returns:

#     """
#     results = {}
#


#     for model in models:
#         total_inf_time = 0
#         total_gpu_mem = 0
#         file_count = 0
#

#

#         for seed in seeds:

#             for fold_idx in range(24):
#                 fold_name = f"fold{fold_idx}"
#                 file_path = os.path.join(base_path, seed, model, fold_name, "inference_stats_test_full.json")
#

#                 if os.path.exists(file_path):
#                     try:
#                         with open(file_path, 'r') as f:
#                             data = json.load(f)

#                             total_inf_time += data.get("avg_inf_time_ms", 0)
#                             total_gpu_mem += data.get("avg_gpu_mem_mb", 0)
#                             file_count += 1
#                     except json.JSONDecodeError:

#                     except KeyError:

#                 else:

#

#         if file_count > 0:
#             avg_inf_time = total_inf_time / file_count
#             avg_gpu_mem = total_gpu_mem / file_count
#             results[model] = {
#                 "average_inference_time_ms": avg_inf_time,
#                 "average_gpu_memory_mb": avg_gpu_mem,
#                 "files_processed": file_count
#             }

#         else:
#             results[model] = {

#             }

#
#     return results
#
#
# if __name__ == "__main__":


#     base_path = "/home/lipei/project/WSDDN/test_results/WEAR/"

#     seeds = ["2022", "2024", "2026"]

#     models = ["wsddn_01", "oicr_01", "pcl_01"]
#

#     final_results = calculate_model_statistics(base_path, seeds, models)
#

#     print("\n" + "=" * 40)

#     print("=" * 40)
#     for model, data in final_results.items():

#         if "error" in data:

#         else:



#     print("\n" + "=" * 40)
#


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



MODEL_DATA = {
    'DCASE': 1180514,
    'CDur': 374163,
    'WSDNN': 4226620,
    'OICR': 31907948,
    'PCL': 31907948,
    'RSKP': 274944,
    'CoLA': 3992454,
}




# def plot_lollipop_chart(data):

#     df = pd.DataFrame(list(data.items()), columns=['Model', 'Parameters'])
#     df = df.sort_values(by='Parameters', ascending=True)
#
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(12, 4))
#

#     ax.hlines(y=df['Model'], xmin=0, xmax=df['Parameters'], color='skyblue', alpha=0.6, linewidth=2)

#     ax.scatter(df['Parameters'], df['Model'], color='dodgerblue', s=150, alpha=1, zorder=3)
#

#     for index, row in df.iterrows():
#         label = f'{row["Parameters"] / 1e6:.1f}M' if row["Parameters"] >= 1e6 else f'{row["Parameters"] / 1e3:.1f}K'
#         ax.text(row['Parameters'] + 800000, row['Model'], label, color='black', ha='left', va='center')
#

#     # ax.set_title('Model Parameters', fontsize=18, fontweight='bold', pad=20)
#     # ax.set_xlabel('Number of Parameters', fontsize=14)

#     ax.set_ylabel('')
#     sns.despine(left=True, bottom=True)
#     plt.tight_layout()
#     plt.show()


# def plot_donut_chart(data):

#     df = pd.DataFrame(list(data.items()), columns=['Model', 'Parameters'])
#     df = df.sort_values(by='Parameters', ascending=False)
#

#     top_n = 4
#     df_top = df.iloc[:top_n]
#     others_sum = df.iloc[top_n:]['Parameters'].sum()
#     if others_sum > 0:
#         df_top = pd.concat([df_top, pd.DataFrame([{'Model': 'Others', 'Parameters': others_sum}])], ignore_index=True)
#
#     labels = df_top['Model']
#     sizes = df_top['Parameters']
#

#     explode = [0.05] + [0] * (len(labels) - 1)
#
#     plt.style.use('default')
#     fig, ax = plt.subplots(figsize=(10, 10))
#

#     colors = sns.color_palette('pastel')
#
#     wedges, texts, autotexts = ax.pie(
#         sizes, labels=labels, autopct='%1.1f%%',
#         startangle=90, pctdistance=0.85, explode=explode,
#         colors=colors, textprops={'fontsize': 12}
#     )
#

#     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#     ax.add_artist(centre_circle)
#

#     ax.axis('equal')
#     ax.set_title('Proportion of Model Parameters', fontsize=18, fontweight='bold', pad=20)
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_minimalist_barchart(data):

#     df = pd.DataFrame(list(data.items()), columns=['Model', 'Parameters'])
#     df['Parameters_M'] = df['Parameters'] / 1_000_000
#     df = df.sort_values(by='Parameters', ascending=True)
#
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(12, 7))
#

#     colors = ['lightgray'] * (len(df) - 2) + ['cornflowerblue'] * 2
#     bars = ax.barh(df['Model'], df['Parameters_M'], color=colors)
#

#     ax.grid(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#

#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')

#

#     for bar in bars:
#         width = bar.get_width()
#         label_text = f'{width:.1f} M'
#         ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, label_text, va='center', ha='left', fontsize=11,
#                 color='black')
#

#     ax.tick_params(axis='y', labelsize=14, labelcolor='black')
#

#     fig.text(0.125, 0.95, 'OICR & PCL Dominate in Model Size', fontsize=20, fontweight='bold', ha='left')
#     fig.text(0.125, 0.90, 'Comparison of model parameters in millions (M)', fontsize=14, ha='left', color='dimgray')
#

#     plt.show()


def plot_grouped_color_lollipop_chart(
        data,
        min_circle_size=60,
        max_circle_size=600,
        label_offset_points=15,
        figsize=(12, 8)
):
    """
    MessageLaTeXMessage，Message。
    """



    color_group_map = {

        'DCASE': '#90EE90',
        'CDur': '#90EE90',


        'WSDNN': '#FFC069',
        'OICR': '#FFC069',
        'PCL': '#FFC069',


        'RSKP': '#FF8787',
        'CoLA': '#FF8787',
    }


    df = pd.DataFrame(list(data.items()), columns=['Model', 'Parameters'])
    df = df.sort_values(by='Parameters', ascending=True)


    assigned_colors = [color_group_map.get(model, '#a9a9a9') for model in df['Model']]


    log_params = np.log10(df['Parameters'])
    min_log, max_log = log_params.min(), log_params.max()
    if max_log == min_log:
        normalized_sizes = np.full_like(log_params, min_circle_size)
    else:
        normalized_sizes = min_circle_size + \
                           (log_params - min_log) / (max_log - min_log) * (max_circle_size - min_circle_size)


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)


    ax.hlines(y=df['Model'], xmin=0, xmax=df['Parameters'], colors=assigned_colors, alpha=0.6, linewidth=2)
    ax.scatter(df['Parameters'], df['Model'], s=normalized_sizes, c=assigned_colors, alpha=1, zorder=3)


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



if __name__ == "__main__":

    plot_grouped_color_lollipop_chart(MODEL_DATA,figsize=(12, 5))


    # plot_donut_chart(MODEL_DATA)
    #

    # plot_minimalist_barchart(MODEL_DATA)

