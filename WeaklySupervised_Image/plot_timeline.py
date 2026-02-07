import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_gt_segments(gt_path, video_id, subset="test"):
    """
    兼容 ActivityNet 风格 GT:
    data["database"][video_id]["annotations"] 每条包含:
      - label: str
      - segment: [t0, t1]  (通常是秒；你的Opportunity也按秒)
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    info = gt["database"][video_id]
    if subset is not None and info.get("subset") != subset:
        # 你也可以这里不强行判断 subset
        pass

    segs = []
    for ann in info.get("annotations", []):
        label = ann.get("label", "unknown")
        t0, t1 = float(ann["segment"][0]), float(ann["segment"][1])
        if t1 > t0:
            segs.append((t0, t1, label))
    return segs

def _load_pred_segments(pred_path, video_id):
    """
    你的 predictions_test_full.json:
    data["results"][video_id] 每条包含:
      - label: str
      - score: float
      - segment: [t0, t1]
    """
    with open(pred_path, "r", encoding="utf-8") as f:
        pred = json.load(f)

    segs = []
    for r in pred["results"].get(video_id, []):
        label = r.get("label", "unknown")
        score = float(r.get("score", 0.0))
        t0, t1 = float(r["segment"][0]), float(r["segment"][1])
        if t1 > t0:
            segs.append((t0, t1, label, score))
    return segs

# 颜色相近
def _build_label2color(labels):
    # 类别多的话 tab20/tab20b/tab20c 都行
    # cmap = plt.get_cmap("tab20")
    # labels = list(labels)
    # label2color = {lab: cmap(i % cmap.N) for i, lab in enumerate(labels)}

    """
        通过合并 tab20, tab20b, tab20c 来创建一个包含 60 种颜色的、
        更多样的调色板。
        """
    # # 1. 分别从三个 colormap 中获取它们的颜色列表
    # #    .colors 属性会返回一个包含所有颜色 (通常是 RGBA 元组) 的列表/元组
    colors_a = plt.get_cmap("tab20").colors
    colors_b = plt.get_cmap("tab20b").colors
    colors_c = plt.get_cmap("tab20c").colors

    # 2. 将这三个列表合并成一个大的颜色池 (20 + 20 + 20 = 60 种颜色)
    #    我们用 list() 转换确保它们是列表，然后用 + 号连接
    color_pool = list(colors_a) + list(colors_b) + list(colors_c)

    # 3. 确保标签有序，以便每次运行的颜色分配都一样
    labels = list(labels)

    # 4. 创建“标签到颜色”的映射字典，但这次是基于新的、更大的颜色池
    label2color = {
        lab: color_pool[i % len(color_pool)] for i, lab in enumerate(labels)
    }

    return label2color

# 不重叠可视化：
def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

def merge_same_label(segs, gap=0.0):
    """同label重叠/相邻(gap内)就合并"""
    by = defaultdict(list)
    for t0,t1,lab,sc in segs:
        by[lab].append((t0,t1,lab,sc))
    out = []
    for lab, items in by.items():
        items.sort(key=lambda x: x[0])
        cur = list(items[0])
        for t0,t1,_,sc in items[1:]:
            if t0 <= cur[1] + gap:  # 重叠或相邻
                cur[1] = max(cur[1], t1)
                cur[3] = max(cur[3], sc)
            else:
                out.append(tuple(cur))
                cur = [t0,t1,lab,sc]
        out.append(tuple(cur))
    out.sort(key=lambda x: x[0])
    return out

def suppress_diff_labels_drop(pred_segs):
    """
    不同label只要有任何重叠：保留分高的，低分整段丢掉（贪心）
    """
    segs = [(float(t0), float(t1), str(lab), float(sc)) for t0,t1,lab,sc in pred_segs if float(t1) > float(t0)]
    segs.sort(key=lambda x: x[3], reverse=True)  # 高分优先
    kept = []
    for t0,t1,lab,sc in segs:
        conflict = any((lab != klab) and overlap((t0,t1),(kt0,kt1)) > 0 for kt0,kt1,klab,ksc in kept)
        if conflict:
            continue
        kept.append((t0,t1,lab,sc))
    # 最后同类合并
    return merge_same_label(kept, gap=0.0)





def plot_action_sequences_frameless(
        gt_segs,
        pred_full_segs,
        pred_window_segs,
        label2color,
        all_labels,
        # --- 绘图控制参数 ---
        title="Action Sequences",
        figsize=(25, 3),
        bar_height=0.4,
        legend_cols=10,
        title_fontsize=20,
        axis_label_fontsize=14,
        tick_label_fontsize=16,  # 这个参数现在用于左侧的文本标签
):
    """
    此版本实现了“无框”效果，去掉了外侧方框，并在左侧手动添加序列名称。
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 1. Y轴位置保持不变，用于定位
    y_positions = {"GT": 2, "Pred_full": 1, "Pred_window": 0}

    # 2. 绘制条形图
    sequences = {"GT": gt_segs, "Pred_full": pred_full_segs, "Pred_window": pred_window_segs}
    all_segs_for_time = gt_segs + pred_full_segs + pred_window_segs
    if not all_segs_for_time:
        plt.close(fig);
        return
    t_min, t_max = min(s[0] for s in all_segs_for_time), max(s[1] for s in all_segs_for_time)

    for name, segs in sequences.items():
        y_pos = y_positions[name]
        for seg in segs:
            color = label2color.get(seg[2], 'gray')
            ax.broken_barh([(seg[0], seg[1] - seg[0])], (y_pos - bar_height / 2, bar_height), facecolors=color,
                           edgecolor='white', linewidth=0.5)

    # 3. 配置坐标轴和边框，实现“无框”效果
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-0.5, 2.5)

    # <--- 核心改动：隐藏Y轴刻度和大部分边框 --->
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)  # 只保留并加粗底部X轴

    # <--- 核心改动：在左侧手动添加文本标签 --->
    for name, y_pos in y_positions.items():
        ax.text(
            t_min - (t_max - t_min) * 0.01, y_pos, name,
            va='center', ha='right', fontsize=tick_label_fontsize, fontweight='bold'
        )

    ax.set_xlabel("Time (seconds)", fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # 4. 配置图例
    handles = [mpatches.Patch(color=label2color[lab], label=lab) for lab in all_labels]
    num_rows = math.ceil(len(all_labels) / legend_cols)
    fig.tight_layout(rect=[0, 0.05 * num_rows, 1, 0.95])

    ax.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1 - (0.08 * num_rows)),
        ncol=legend_cols,
        fontsize=12,
        frameon=False
    )

    plt.show()


# ==============================================================================
# 主逻辑
# ==============================================================================

# --- 1. 路径和ID ---
# WETLAB
# gt_path = "/home/lipei/project/WSDDN/fold0/gt_for_anet.json"
# pred_full_path = "/home/lipei/project/WSDDN/fold0/predictions_test_full.json"
# pred_window_path = "/home/lipei/project/WSDDN/fold0/predictions_test_window.json"

# WEAR
# gt_path = "/home/lipei/project/WSDDN/test_results/WEAR/2026/rskp_0126/fold0/gt_for_anet.json"
# pred_full_path =  "/home/lipei/project/WSDDN/test_results/WEAR/2026/rskp_0126/fold0/predictions_test_full.json"
# pred_window_path = "/home/lipei/project/WSDDN/test_results/WEAR/2026/rskp_0126/fold0/predictions_test_window.json"

# SBHAR
# gt_path = "/home/lipei/project/WSDDN/test_results/SBHAR/2026/rskp_0125/fold0/gt_for_anet.json"
# pred_full_path = "/home/lipei/project/WSDDN/test_results/SBHAR/2026/rskp_0125/fold0/predictions_test_full.json"
# pred_window_path = "/home/lipei/project/WSDDN/test_results/SBHAR/2026/rskp_0125/fold0/predictions_test_window.json"

# Opportunity
gt_path = "/home/lipei/project/WSDDN/test_results/Opportunity/pcl_0106/fold0/gt_for_anet.json"
pred_full_path = "/home/lipei/project/WSDDN/test_results/Opportunity/pcl_0106/fold0/predictions_test_full.json"
pred_window_path = "/home/lipei/project/WSDDN/test_results/Opportunity/pcl_0106/fold0/predictions_test_full.json"

# RWHAR
# gt_path = "/home/lipei/project/WSDDN/test_results/RWHAR/wsddn_0105/fold0/gt_for_anet.json"
# pred_full_path = "/home/lipei/project/WSDDN/test_results/Opportunity/pcl_0106/fold0/predictions_test_full.json"
# pred_window_path = "/home/lipei/project/WSDDN/test_results/Opportunity/pcl_0106/fold0/predictions_test_full.json"



video_id = "sbj_0"

# --- 2. 加载和处理数据 ---
try:
    gt_segs = _load_gt_segments(gt_path, video_id)
    pred_full_segs_raw = _load_pred_segments(pred_full_path, video_id)
    pred_window_segs_raw = _load_pred_segments(pred_window_path, video_id)
except Exception:
    gt_segs, pred_full_segs_raw, pred_window_segs_raw = [], [], []


topk_total = 200
pred_full_segs = suppress_diff_labels_drop(sorted(pred_full_segs_raw, key=lambda x: x[3], reverse=True)[:topk_total])
pred_window_segs = suppress_diff_labels_drop(
    sorted(pred_window_segs_raw, key=lambda x: x[3], reverse=True)[:topk_total])

# --- 3. 准备颜色和标签 ---
all_labels = sorted({s[2] for s in gt_segs} | {s[2] for s in pred_full_segs} | {s[2] for s in pred_window_segs})
label2color = _build_label2color(all_labels)

# --- 4. 调用新的“无框”绘图函数 ---
if all_labels:
    plot_action_sequences_frameless(
        gt_segs, pred_full_segs, pred_window_segs,
        label2color, all_labels,
        title="SBHAR--RSKP",
        figsize=(20, 2.70),
        bar_height=0.6,
        legend_cols=8,
        title_fontsize=18,
        axis_label_fontsize=12,
        tick_label_fontsize=14,  # 控制左侧 "GT", "Pred_full" 等文本的大小
    )
else:
    print(f"在 video_id '{video_id}' 中没有找到任何要绘制的数据。")




# def _load_json(path: str):
#     """安全地加载JSON文件。"""
#     if not os.path.exists(path):
#         print(f"警告：文件未找到 -> {path}")
#         return None
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except json.JSONDecodeError:
#         print(f"错误：文件 '{path}' 不是有效的JSON格式。")
#         return None
#
#
# def _load_gt_segments(gt_path, video_id):
#     """
#     加载Ground Truth数据。此函数已兼容您的新GT格式。
#     """
#     gt_data = _load_json(gt_path)
#     if not gt_data or "database" not in gt_data or video_id not in gt_data["database"]:
#         print(f"警告：无法从 '{gt_path}' 加载 video_id '{video_id}' 的GT数据。")
#         return []
#     info = gt_data["database"][video_id]
#     return [(float(ann["segment"][0]), float(ann["segment"][1]), ann.get("label", "unknown")) for ann in
#             info.get("annotations", []) if float(ann["segment"][1]) > float(ann["segment"][0])]
#
#
# def _load_pred_segments(pred_path, video_id):
#     """
#     加载预测结果。此函数已兼容您的新Pred格式。
#     """
#     pred_data = _load_json(pred_path)
#     if not pred_data or "results" not in pred_data or video_id not in pred_data["results"]:
#         print(f"警告：无法从 '{pred_path}' 加载 video_id '{video_id}' 的预测数据。")
#         return []
#     return [(float(r["segment"][0]), float(r["segment"][1]), r.get("label", "unknown"), float(r.get("score", 0.0))) for
#             r in pred_data["results"].get(video_id, []) if float(r["segment"][1]) > float(r["segment"][0])]
#
#
# def _build_label2color(labels):
#     """创建从标签到颜色的映射。"""
#     color_pool = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(
#         plt.get_cmap("tab20c").colors)
#     return {lab: color_pool[i % len(color_pool)] for i, lab in enumerate(sorted(list(labels)))}
#
#
# def overlap(a, b): return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
#
#
# def merge_same_label(segs, gap=0.0):
#     """合并相同标签的邻近或重叠段。"""
#     by, out = defaultdict(list), []
#     for t0, t1, lab, sc in segs: by[lab].append((t0, t1, lab, sc))
#     for lab, items in by.items():
#         items.sort(key=lambda x: x[0])
#         cur = list(items[0])
#         for t0, t1, _, sc in items[1:]:
#             if t0 <= cur[1] + gap:
#                 cur[1], cur[3] = max(cur[1], t1), max(cur[3], sc)
#             else:
#                 out.append(tuple(cur));
#                 cur = [t0, t1, lab, sc]
#         out.append(tuple(cur))
#     return sorted(out, key=lambda x: x[0])
#
#
# def suppress_diff_labels_drop(pred_segs):
#     """NMS：对于重叠的不同标签，保留得分高的。"""
#     segs = sorted(pred_segs, key=lambda x: x[3], reverse=True)
#     kept = []
#     for t0, t1, lab, sc in segs:
#         if not any((lab != klab) and overlap((t0, t1), (kt0, kt1)) > 0 for kt0, kt1, klab, ksc in kept):
#             kept.append((t0, t1, lab, sc))
#     return merge_same_label(kept, gap=0.0)
#
#
# def plot_action_sequences_frameless(
#         gt_segs, pred_full_segs, pred_window_segs,
#         label2color, all_labels,
#         title, figsize, bar_height, legend_cols,
#         title_fontsize, axis_label_fontsize, tick_label_fontsize
# ):
#     """
#     “无框”风格的最终绘图函数。
#     """
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     y_positions = {"GT": 2, "Pred_full": 1, "Pred_window": 0}
#
#     all_segs_for_time = gt_segs + pred_full_segs + pred_window_segs
#     if not all_segs_for_time:
#         print("没有找到任何用于绘图的数据段。")
#         plt.close(fig);
#         return
#     t_min, t_max = min(s[0] for s in all_segs_for_time), max(s[1] for s in all_segs_for_time)
#
#     # 保证t_min从0开始，如果数据不是从0开始的话
#     if t_min > 0:
#         t_min = 0
#
#     sequences = {"GT": gt_segs, "Pred_full": pred_full_segs, "Pred_window": pred_window_segs}
#     for name, segs in sequences.items():
#         y_pos = y_positions[name]
#         for seg in segs:
#             color = label2color.get(seg[2], 'gray')
#             ax.broken_barh([(seg[0], seg[1] - seg[0])], (y_pos - bar_height / 2, bar_height), facecolors=color,
#                            edgecolor='white', linewidth=0.5)
#
#     ax.set_xlim(t_min, t_max)
#     ax.set_ylim(-0.5, 2.5)
#     ax.set_yticks([])
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(1.2)
#
#     text_x_pos = t_min - (t_max - t_min) * 0.01  # 计算左侧文字的X坐标
#     for name, y_pos in y_positions.items():
#         ax.text(text_x_pos, y_pos, name, va='center', ha='right', fontsize=tick_label_fontsize, fontweight='bold')
#
#     ax.set_xlabel("Time (frames)", fontsize=axis_label_fontsize)  # 注意：这里可能是帧数
#     ax.set_title(title, fontsize=title_fontsize)
#     ax.grid(True, axis="x", linestyle="--", alpha=0.5)
#
#     handles = [mpatches.Patch(color=label2color[lab], label=lab) for lab in all_labels]
#     num_rows = math.ceil(len(all_labels) / legend_cols)
#     fig.tight_layout(rect=[0, 0.05 * num_rows, 1, 0.95])
#
#     ax.legend(
#         handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12 - (0.08 * num_rows)),
#         ncol=legend_cols, fontsize=12, frameon=False
#     )
#     plt.show()
#
#
# # ==============================================================================
# # == 主逻辑：适配您的新数据集 ==
# # ==============================================================================
# if __name__ == "__main__":
#     # --- 1. 请在这里配置您的新数据集文件路径 ---
#     GT_PATH = "/home/lipei/XRFV2/imu_annotations.json"
#     PRED_FULL_PATH = "/home/lipei/project/WSDDN/test_results/xrfv2/2026/xrfv2_CNN_rskp_person6/predictions_test_full.json"
#     PRED_WINDOW_PATH = "/home/lipei/project/WSDDN/test_results/xrfv2/2026/xrfv2_CNN_rskp_person6/predictions_test_window.json"
#
#     # --- 2. 请指定您想可视化的 video_id ---
#     # 例如： "6_1_10.h5", "0_1_3.h5" 等
#     VIDEO_ID_TO_PLOT = "6_1_5.h5"
#
#     # --- 3. (可选) 可以在这里设置过滤参数 ---
#     TOP_K = 200  # 保留分数最高的K个预测
#     SCORE_THRESHOLD = None  # 或 0.01, 过滤掉分数低于此值的预测
#
#     # --- 4. 数据加载与处理 ---
#     print(f"正在加载 video_id: {VIDEO_ID_TO_PLOT}...")
#     gt_segments = _load_gt_segments(GT_PATH, VIDEO_ID_TO_PLOT)
#     pred_full_raw = _load_pred_segments(PRED_FULL_PATH, VIDEO_ID_TO_PLOT)
#     pred_window_raw = _load_pred_segments(PRED_WINDOW_PATH, VIDEO_ID_TO_PLOT)
#
#     # 应用过滤
#     if SCORE_THRESHOLD is not None:
#         pred_full_raw = [s for s in pred_full_raw if s[3] >= SCORE_THRESHOLD]
#         pred_window_raw = [s for s in pred_window_raw if s[3] >= SCORE_THRESHOLD]
#
#     if TOP_K is not None:
#         pred_full_raw = sorted(pred_full_raw, key=lambda x: x[3], reverse=True)[:TOP_K]
#         pred_window_raw = sorted(pred_window_raw, key=lambda x: x[3], reverse=True)[:TOP_K]
#
#     # 应用NMS
#     pred_full_segments = suppress_diff_labels_drop(pred_full_raw)
#     pred_window_segments = suppress_diff_labels_drop(pred_window_raw)
#
#     # --- 5. 准备标签和颜色 ---
#     all_action_labels = sorted(
#         {s[2] for s in gt_segments} |
#         {s[2] for s in pred_full_segments} |
#         {s[2] for s in pred_window_segments}
#     )
#     color_map = _build_label2color(all_action_labels)
#
#     # --- 6. 调用绘图函数 ---
#     if all_action_labels:
#         print("数据加载完毕，开始绘图...")
#         plot_action_sequences_frameless(
#             gt_segs=gt_segments,
#             pred_full_segs=pred_full_segments,
#             pred_window_segs=pred_window_segments,
#             label2color=color_map,
#             all_labels=all_action_labels,
#             # --- 您可以在这里自定义绘图的细节 ---
#             title=f"XRFV2--RSKP",
#             figsize=(20, 2.5),
#             bar_height=0.4,
#             legend_cols=6,
#             title_fontsize=18,
#             axis_label_fontsize=12,
#             tick_label_fontsize=14
#         )
#     else:
#         print(f"错误：在所有文件中均未找到关于 video_id '{VIDEO_ID_TO_PLOT}' 的有效数据，无法绘图。")
