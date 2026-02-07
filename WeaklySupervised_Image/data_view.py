# # import json
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# # import os
# #
# #
# # def plot_sbj0_action_sequence(file_path, dataset_name="Opportunity"):
# #     if not os.path.exists(file_path):
# #
# #     with open(file_path, 'r', encoding='utf-8') as f:
# #         data = json.load(f)
# #
# #     try:
# #         sbj0_data = data['database']['sbj_0']
# #         total_duration = sbj0_data['duration']
# #         annotations = sbj0_data['annotations']
# #     except KeyError as e:
# #
# #     base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
# #                    '#17becf']
# #     unique_labels = list(set([ann['label'] for ann in annotations]))
# #     color_map = {label: base_colors[i % len(base_colors)] for i, label in enumerate(unique_labels)}
# #
# #     fig, ax = plt.subplots(figsize=(20, 2.5))
# #     ax.set_xlim(0, total_duration)
# #     ax.set_ylim(0, 1)
# #     ax.set_yticks([])
# #     ax.spines['left'].set_visible(False)
# #     ax.spines['right'].set_visible(False)
# #     ax.spines['top'].set_visible(False)
# #     ax.spines['bottom'].set_linewidth(1)
# #
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
# #     ax.text(
# #         -total_duration * 0.01, 0.5, dataset_name,
# #         va='center', ha='right', fontsize=16, fontweight='bold'
# #     )
# #
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
    filesbj_0action
    :param file_path: fileJSONpath
    :param dataset_name: dataset（）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file，path：{file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        sbj0_data = data['database']['sbj_0']
        total_duration = sbj0_data['duration']
        annotations = sbj0_data['annotations']
    except KeyError as e:
        raise KeyError(f"，：{e}，database/sbj_0/duration/annotations")

    medium_soft_colors = [
        '#73a8ff', '#85e89d', '#fff275', '#ffab70', '#ff8787',
        '#c098ff', '#71c6dd', '#ffd470', '#71dd71', '#a6a6a6',
        '#95de64', '#ffc069'
    ]
    unique_labels = list(set([ann['label'] for ann in annotations]))
    color_map = {label: medium_soft_colors[i % len(medium_soft_colors)] for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(20, 2.5))
    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)

    for ann in annotations:
        label = ann['label']
        start, end = ann['segment']
        rect_width = end - start
        rect = patches.Rectangle(
            (start, 0.2), rect_width, 0.6,
            color=color_map[label],
            linewidth=0.5,
            edgecolor='white',
            zorder=2
        )
        ax.add_patch(rect)

    ax.text(
        -total_duration * 0.01, 0.5, dataset_name,
        va='center', ha='right', fontsize=16, fontweight='bold',
        fontfamily='SimHei'
    )

    ax.set_xlabel("Time (seconds)", fontsize=14, labelpad=12, fontweight='medium')

    plt.tight_layout()
    # plt.savefig(f"{dataset_name}_action_sequence.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


if __name__ == "__main__":
    DATA_FILE_PATH = r"data/hangtime/annotations/loso_sbj_0.json"
    plot_sbj0_action_sequence(DATA_FILE_PATH, dataset_name="Hang-Time")


# import json
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import os
#
#
# def plot_013h5_action_sequence(file_path, dataset_name="0_1_3.h5", sample_rate=50):
#     """
#     """
#     if not os.path.exists(file_path):
#
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     try:
#         target_data = data['database']['0_1_3.h5']
#         annotations = target_data['annotations']
#     except KeyError as e:
#
#     anns_seconds = []
#     for ann in annotations:
#         start_f, end_f = ann['segment']
#         start_s = round(start_f / sample_rate, 2)
#         end_s = round(end_f / sample_rate, 2)
#         anns_seconds.append({
#             "label": ann['label'],
#             "segment_s": [start_s, end_s]
#         })
#     total_duration = max(ann['segment_s'][1] for ann in anns_seconds)
#
#     medium_soft_colors = [
#         '#73a8ff', '#85e89d', '#fff275', '#ffab70', '#ff8787',
#         '#c098ff', '#71c6dd', '#ffd470', '#71dd71', '#a6a6a6',
#         '#95de64', '#ffc069'
#     ]
#     unique_labels = list(set([ann['label'] for ann in annotations]))
#     color_map = {label: medium_soft_colors[i % len(medium_soft_colors)] for i, label in enumerate(unique_labels)}
#
#     fig, ax = plt.subplots(figsize=(20, 2.5))
#
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(1.2)
#
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
#     ax.text(
#         -total_duration * 0.01, 0.5, dataset_name,
#         va='center', ha='right', fontsize=16, fontweight='bold',
#         fontfamily='SimHei'
#     )
#
#     ax.set_xlabel("Time (seconds)", fontsize=14, labelpad=12, fontweight='medium')
#
#     plt.tight_layout()
#     # plt.savefig(f"{dataset_name}_action_sequence.png", dpi=300, bbox_inches='tight', facecolor='white')
#     plt.show()
#
#
# if __name__ == "__main__":
#     DATA_FILE_PATH = r"/home/lipei/XRFV2/imu_annotations.json"
#     plot_013h5_action_sequence(DATA_FILE_PATH, dataset_name="XRFV2")
