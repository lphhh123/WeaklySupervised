# # import json
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# # import os
# #
# #
# # def plot_sbj0_action_sequence(file_path, dataset_name="Opportunity"):
# #     if not os.path.exists(file_path):
# #         raise FileNotFoundError(f"文件不存在，请检查路径：{file_path}")
# #
# #     with open(file_path, 'r', encoding='utf-8') as f:
# #         data = json.load(f)
# #
# #     try:
# #         sbj0_data = data['database']['sbj_0']
# #         total_duration = sbj0_data['duration']
# #         annotations = sbj0_data['annotations']
# #     except KeyError as e:
# #         raise KeyError(f"文件数据格式错误，缺少关键字段：{e}")
# #
# #     # 颜色映射（仅用于绘图，不生成图例）
# #     base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
# #                    '#17becf']
# #     unique_labels = list(set([ann['label'] for ann in annotations]))
# #     color_map = {label: base_colors[i % len(base_colors)] for i, label in enumerate(unique_labels)}
# #
# #     # 创建画布
# #     fig, ax = plt.subplots(figsize=(20, 2.5))
# #     ax.set_xlim(0, total_duration)
# #     ax.set_ylim(0, 1)
# #     ax.set_yticks([])
# #     # 隐藏所有边框，只保留X轴
# #     ax.spines['left'].set_visible(False)
# #     ax.spines['right'].set_visible(False)
# #     ax.spines['top'].set_visible(False)
# #     ax.spines['bottom'].set_linewidth(1)
# #
# #     # 绘制动作块
# #     for ann in annotations:
# #         label = ann['label']
# #         start, end = ann['segment']
# #         rect_width = end - start
# #         rect = patches.Rectangle(
# #             (start, 0.2), rect_width, 0.6,
# #             color=color_map[label],
# #             alpha=0.9,
# #             linewidth=0.3, edgecolor='white'
# #         )
# #         ax.add_patch(rect)
# #
# #     # 左侧添加数据集名称
# #     ax.text(
# #         -total_duration * 0.01, 0.5, dataset_name,
# #         va='center', ha='right', fontsize=16, fontweight='bold'
# #     )
# #
# #     # X轴标签
# #     ax.set_xlabel("Time (seconds)", fontsize=14, labelpad=10)
# #     plt.tight_layout()
# #     # plt.savefig("sbj0_action_sequence_no_legend.png", dpi=300, bbox_inches='tight')
# #     plt.show()
# #
#
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def plot_sbj0_action_sequence(file_path, dataset_name="Opportunity"):
    """
    可视化指定文件中sbj_0的动作序列
    :param file_path: 数据文件JSON路径
    :param dataset_name: 左侧显示的数据集名称（自定义）
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在，请检查路径：{file_path}")

    # 读取并解析JSON数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取sbj_0核心数据（异常捕获，防止字段缺失）
    try:
        sbj0_data = data['database']['sbj_0']
        total_duration = sbj0_data['duration']
        annotations = sbj0_data['annotations']
    except KeyError as e:
        raise KeyError(f"数据格式错误，缺少关键字段：{e}，请检查database/sbj_0/duration/annotations")

    # ---------------------- 核心调整：适中浅色系（辨识度高，不厚重，12种可循环） ----------------------
    medium_soft_colors = [
        '#73a8ff', '#85e89d', '#fff275', '#ffab70', '#ff8787',
        '#c098ff', '#71c6dd', '#ffd470', '#71dd71', '#a6a6a6',
        '#95de64', '#ffc069'
    ]
    # 提取唯一动作标签，分配配色
    unique_labels = list(set([ann['label'] for ann in annotations]))
    color_map = {label: medium_soft_colors[i % len(medium_soft_colors)] for i, label in enumerate(unique_labels)}

    # 创建画布（适配长序列，尺寸经典）
    fig, ax = plt.subplots(figsize=(20, 2.5))
    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])  # 隐藏Y轴刻度

    # 隐藏无用边框，仅保留X轴（加粗，提升层次感）
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)

    # 批量绘制动作块（白色细边框，区分相邻/重叠块）
    for ann in annotations:
        label = ann['label']
        start, end = ann['segment']
        rect_width = end - start
        rect = patches.Rectangle(
            (start, 0.2), rect_width, 0.6,
            color=color_map[label],  # 适中浅色系填充
            linewidth=0.5,  # 细边框宽度
            edgecolor='white',  # 白色边框提升区分度
            zorder=2  # 图层优先级，避免被遮挡
        )
        ax.add_patch(rect)

    # 左侧添加数据集名称（加粗，位置适中，兼容中文）
    ax.text(
        -total_duration * 0.01, 0.5, dataset_name,
        va='center', ha='right', fontsize=16, fontweight='bold',
        fontfamily='SimHei'  # 无需中文可删除此参数
    )

    # X轴标签（加粗，间距适中）
    ax.set_xlabel("Time (seconds)", fontsize=14, labelpad=12, fontweight='medium')

    # 调整布局，防止边缘内容被裁剪
    plt.tight_layout()
    # 可选：保存高清白底图片（直接用于报告/论文，取消注释即可）
    # plt.savefig(f"{dataset_name}_action_sequence.png", dpi=300, bbox_inches='tight', facecolor='white')
    # 显示可视化图
    plt.show()


if __name__ == "__main__":
    DATA_FILE_PATH = r"/home/lipei/TAL_data/hangtime/annotations/loso_sbj_0.json"
    plot_sbj0_action_sequence(DATA_FILE_PATH, dataset_name="Hang-Time")


# import json
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import os
#
#
# def plot_013h5_action_sequence(file_path, dataset_name="0_1_3.h5", sample_rate=50):
#     """
#     可视化新格式数据集中0_1_3.h5的动作序列（帧数转秒数，统一X轴为秒）
#     :param file_path: 数据文件JSON路径
#     :param dataset_name: 左侧显示的数据集名称（默认0_1_3.h5）
#     :param sample_rate: 采样率（固定50，无需修改）
#     """
#     # 检查文件是否存在
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"文件不存在，请检查路径：{file_path}")
#
#     # 读取并解析JSON数据
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     # 提取0_1_3.h5的核心数据（异常捕获，防止字段缺失）
#     try:
#         target_data = data['database']['0_1_3.h5']
#         annotations = target_data['annotations']
#     except KeyError as e:
#         raise KeyError(f"数据格式错误，缺少关键字段：{e}，请检查database/0_1_3.h5/annotations")
#
#     # ---------------------- 核心：帧数转秒数（采样率50） ----------------------
#     anns_seconds = []
#     for ann in annotations:
#         start_f, end_f = ann['segment']
#         # 帧数 / 采样率 = 秒数，保留2位小数更整洁
#         start_s = round(start_f / sample_rate, 2)
#         end_s = round(end_f / sample_rate, 2)
#         anns_seconds.append({
#             "label": ann['label'],
#             "segment_s": [start_s, end_s]
#         })
#     # 计算总时长（秒）
#     total_duration = max(ann['segment_s'][1] for ann in anns_seconds)
#
#     # 统一使用中度柔和色系（和sbj_0可视化风格一致，辨识度高）
#     medium_soft_colors = [
#         '#73a8ff', '#85e89d', '#fff275', '#ffab70', '#ff8787',
#         '#c098ff', '#71c6dd', '#ffd470', '#71dd71', '#a6a6a6',
#         '#95de64', '#ffc069'
#     ]
#     # 提取唯一动作标签并分配颜色
#     unique_labels = list(set([ann['label'] for ann in annotations]))
#     color_map = {label: medium_soft_colors[i % len(medium_soft_colors)] for i, label in enumerate(unique_labels)}
#
#     # 创建画布（和之前一致的尺寸，保证可视化风格统一）
#     fig, ax = plt.subplots(figsize=(20, 2.5))
#     ax.set_xlim(0, total_duration)  # X轴：0到总时长（秒）
#     ax.set_ylim(0, 1)  # Y轴：单行动作序列
#     ax.set_yticks([])  # 隐藏Y轴刻度
#
#     # 隐藏无用边框，仅保留X轴（加粗，提升层次感）
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(1.2)
#
#     # 批量绘制动作块（白色细边框，区分相邻/重叠块，无图例无遮挡）
#     for ann in anns_seconds:
#         label = ann['label']
#         start, end = ann['segment_s']
#         rect_width = end - start
#         rect = patches.Rectangle(
#             (start, 0.2), rect_width, 0.6,
#             color=color_map[label],
#             linewidth=0.5,
#             edgecolor='white',
#             zorder=2
#         )
#         ax.add_patch(rect)
#
#     # 左侧添加数据集名称（加粗，位置和之前一致，兼容中文）
#     ax.text(
#         -total_duration * 0.01, 0.5, dataset_name,
#         va='center', ha='right', fontsize=16, fontweight='bold',
#         fontfamily='SimHei'
#     )
#
#     # X轴标签统一为 Time (seconds)，和sbj_0可视化完全一致
#     ax.set_xlabel("Time (seconds)", fontsize=14, labelpad=12, fontweight='medium')
#
#     # 调整布局，防止边缘内容被裁剪
#     plt.tight_layout()
#     # 可选：保存高清白底图片（取消注释即可，文件名自动带数据集名称）
#     # plt.savefig(f"{dataset_name}_action_sequence.png", dpi=300, bbox_inches='tight', facecolor='white')
#     # 显示可视化图
#     plt.show()
#
#
# # ---------------------- 主程序：仅需修改文件路径 ----------------------
# if __name__ == "__main__":
#     # 替换成你的JSON文件实际路径
#     DATA_FILE_PATH = r"/home/lipei/XRFV2/imu_annotations.json"
#     plot_013h5_action_sequence(DATA_FILE_PATH, dataset_name="XRFV2")