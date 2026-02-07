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
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    info = gt["database"][video_id]
    if subset is not None and info.get("subset") != subset:

        pass

    segs = []
    for ann in info.get("annotations", []):
        label = ann.get("label", "unknown")
        t0, t1 = float(ann["segment"][0]), float(ann["segment"][1])
        if t1 > t0:
            segs.append((t0, t1, label))
    return segs

def _load_pred_segments(pred_path, video_id):
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

def _build_label2color(labels):

    # cmap = plt.get_cmap("tab20")
    # labels = list(labels)
    # label2color = {lab: cmap(i % cmap.N) for i, lab in enumerate(labels)}



    colors_a = plt.get_cmap("tab20").colors
    colors_b = plt.get_cmap("tab20b").colors
    colors_c = plt.get_cmap("tab20c").colors



    color_pool = list(colors_a) + list(colors_b) + list(colors_c)


    labels = list(labels)


    label2color = {
        lab: color_pool[i % len(color_pool)] for i, lab in enumerate(labels)
    }

    return label2color

def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

def merge_same_label(segs, gap=0.0):
    by = defaultdict(list)
    for t0,t1,lab,sc in segs:
        by[lab].append((t0,t1,lab,sc))
    out = []
    for lab, items in by.items():
        items.sort(key=lambda x: x[0])
        cur = list(items[0])
        for t0,t1,_,sc in items[1:]:
            if t0 <= cur[1] + gap:
                cur[1] = max(cur[1], t1)
                cur[3] = max(cur[3], sc)
            else:
                out.append(tuple(cur))
                cur = [t0,t1,lab,sc]
        out.append(tuple(cur))
    out.sort(key=lambda x: x[0])
    return out

def suppress_diff_labels_drop(pred_segs):
    segs = [(float(t0), float(t1), str(lab), float(sc)) for t0,t1,lab,sc in pred_segs if float(t1) > float(t0)]
    segs.sort(key=lambda x: x[3], reverse=True)
    kept = []
    for t0,t1,lab,sc in segs:
        conflict = any((lab != klab) and overlap((t0,t1),(kt0,kt1)) > 0 for kt0,kt1,klab,ksc in kept)
        if conflict:
            continue
        kept.append((t0,t1,lab,sc))

    return merge_same_label(kept, gap=0.0)





def plot_action_sequences_frameless(
        gt_segs,
        pred_full_segs,
        pred_window_segs,
        label2color,
        all_labels,

        title="Action Sequences",
        figsize=(25, 3),
        bar_height=0.4,
        legend_cols=10,
        title_fontsize=20,
        axis_label_fontsize=14,
        tick_label_fontsize=16,
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)


    y_positions = {"GT": 2, "Pred_full": 1, "Pred_window": 0}


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


    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-0.5, 2.5)


    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)


    for name, y_pos in y_positions.items():
        ax.text(
            t_min - (t_max - t_min) * 0.01, y_pos, name,
            va='center', ha='right', fontsize=tick_label_fontsize, fontweight='bold'
        )

    ax.set_xlabel("Time (seconds)", fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)


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
# ==============================================================================

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
gt_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/rwhar_dcase_10_500s_2022/fold0/gt_for_anet.json"
pred_full_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/rwhar_dcase_10_500s_2022/fold0/best_predictions_test_full_500s.json"
pred_window_path = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/rwhar_dcase_10_500s_2022/fold0/best_predictions_test_window_500s.json"

# RWHAR
# gt_path = "/home/lipei/project/WSDDN/test_results/RWHAR/wsddn_0105/fold0/gt_for_anet.json"
# pred_full_path = "/home/lipei/project/WSDDN/test_results/Opportunity/pcl_0106/fold0/predictions_test_full.json"
# pred_window_path = "/home/lipei/project/WSDDN/test_results/Opportunity/pcl_0106/fold0/predictions_test_full.json"



video_id = "sbj_0"

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

all_labels = sorted({s[2] for s in gt_segs} | {s[2] for s in pred_full_segs} | {s[2] for s in pred_window_segs})
label2color = _build_label2color(all_labels)

if all_labels:
    plot_action_sequences_frameless(
        gt_segs, pred_full_segs, pred_window_segs,
        label2color, all_labels,
        title="RWHAR--DCASE",
        figsize=(20, 3.40),
        bar_height=0.6,
        legend_cols=8,
        title_fontsize=18,
        axis_label_fontsize=12,
        tick_label_fontsize=14,
    )
else:
    print(f"在 video_id '{video_id}' 中没有找到任何要绘制的数据。")




# def _load_json(path: str):
#     if not os.path.exists(path):
#         return None
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except json.JSONDecodeError:
#         return None
#
#
# def _load_gt_segments(gt_path, video_id):
#     """
#     """
#     gt_data = _load_json(gt_path)
#     if not gt_data or "database" not in gt_data or video_id not in gt_data["database"]:
#         return []
#     info = gt_data["database"][video_id]
#     return [(float(ann["segment"][0]), float(ann["segment"][1]), ann.get("label", "unknown")) for ann in
#             info.get("annotations", []) if float(ann["segment"][1]) > float(ann["segment"][0])]
#
#
# def _load_pred_segments(pred_path, video_id):
#     """
#     """
#     pred_data = _load_json(pred_path)
#     if not pred_data or "results" not in pred_data or video_id not in pred_data["results"]:
#         return []
#     return [(float(r["segment"][0]), float(r["segment"][1]), r.get("label", "unknown"), float(r.get("score", 0.0))) for
#             r in pred_data["results"].get(video_id, []) if float(r["segment"][1]) > float(r["segment"][0])]
#
#
# def _build_label2color(labels):
#     color_pool = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(
#         plt.get_cmap("tab20c").colors)
#     return {lab: color_pool[i % len(color_pool)] for i, lab in enumerate(sorted(list(labels)))}
#
#
# def overlap(a, b): return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
#
#
# def merge_same_label(segs, gap=0.0):
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
#     """
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#     y_positions = {"GT": 2, "Pred_full": 1, "Pred_window": 0}
#
#     all_segs_for_time = gt_segs + pred_full_segs + pred_window_segs
#     if not all_segs_for_time:
#         plt.close(fig);
#         return
#     t_min, t_max = min(s[0] for s in all_segs_for_time), max(s[1] for s in all_segs_for_time)
#
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
#     for name, y_pos in y_positions.items():
#         ax.text(text_x_pos, y_pos, name, va='center', ha='right', fontsize=tick_label_fontsize, fontweight='bold')
#
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
# # ==============================================================================
# if __name__ == "__main__":
#     GT_PATH = "/home/lipei/XRFV2/imu_annotations.json"
#     PRED_FULL_PATH = "/home/lipei/project/WSDDN/test_results/xrfv2/2026/xrfv2_CNN_rskp_person6/predictions_test_full.json"
#     PRED_WINDOW_PATH = "/home/lipei/project/WSDDN/test_results/xrfv2/2026/xrfv2_CNN_rskp_person6/predictions_test_window.json"
#
#     VIDEO_ID_TO_PLOT = "6_1_5.h5"
#
#
#     gt_segments = _load_gt_segments(GT_PATH, VIDEO_ID_TO_PLOT)
#     pred_full_raw = _load_pred_segments(PRED_FULL_PATH, VIDEO_ID_TO_PLOT)
#     pred_window_raw = _load_pred_segments(PRED_WINDOW_PATH, VIDEO_ID_TO_PLOT)
#
#     if SCORE_THRESHOLD is not None:
#         pred_full_raw = [s for s in pred_full_raw if s[3] >= SCORE_THRESHOLD]
#         pred_window_raw = [s for s in pred_window_raw if s[3] >= SCORE_THRESHOLD]
#
#     if TOP_K is not None:
#         pred_full_raw = sorted(pred_full_raw, key=lambda x: x[3], reverse=True)[:TOP_K]
#         pred_window_raw = sorted(pred_window_raw, key=lambda x: x[3], reverse=True)[:TOP_K]
#
#     pred_full_segments = suppress_diff_labels_drop(pred_full_raw)
#     pred_window_segments = suppress_diff_labels_drop(pred_window_raw)
#
#     all_action_labels = sorted(
#         {s[2] for s in gt_segments} |
#         {s[2] for s in pred_full_segments} |
#         {s[2] for s in pred_window_segments}
#     )
#     color_map = _build_label2color(all_action_labels)
#
#     if all_action_labels:
#         plot_action_sequences_frameless(
#             gt_segs=gt_segments,
#             pred_full_segs=pred_full_segments,
#             pred_window_segs=pred_window_segments,
#             label2color=color_map,
#             all_labels=all_action_labels,
#             title=f"XRFV2--RSKP",
#             figsize=(20, 2.5),
#             bar_height=0.4,
#             legend_cols=6,
#             title_fontsize=18,
#             axis_label_fontsize=12,
#             tick_label_fontsize=14
#         )
#     else:
