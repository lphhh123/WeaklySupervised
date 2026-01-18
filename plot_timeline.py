import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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

def _build_label2color(labels):
    # 类别多的话 tab20/tab20b/tab20c 都行
    cmap = plt.get_cmap("tab20")
    labels = list(labels)
    label2color = {lab: cmap(i % cmap.N) for i, lab in enumerate(labels)}
    return label2color

def plot_gt_pred_two_rows_inline(
    gt_path,
    pred_path,
    video_id="sbj_0",
    subset="test",
    # 这两个只是“画图筛选”，不想筛选就设为 None
    score_thresh=None,
    topk_total=None,
    figsize=(18, 4),
):
    gt_segs = _load_gt_segments(gt_path, video_id, subset=subset)

    pred_segs = _load_pred_segments(pred_path, video_id)
    pred_segs = suppress_diff_labels_drop(pred_segs)  # 不重叠可视化

    # ===== 可选：画图时再筛掉一些 pred，避免太密 =====
    if score_thresh is not None:
        pred_segs = [x for x in pred_segs if x[3] >= float(score_thresh)]
    if topk_total is not None:
        pred_segs = sorted(pred_segs, key=lambda x: x[3], reverse=True)[:int(topk_total)]

    # label全集，用于统一配色（GT/Pred 同色）
    all_labels = sorted({lab for _, _, lab in gt_segs} | {lab for _, _, lab, _ in pred_segs})
    label2color = _build_label2color(all_labels)

    # 时间范围
    t_min = min([t0 for t0, _, _ in gt_segs] + [t0 for t0, _, _, _ in pred_segs] + [0.0])
    t_max = max([t1 for _, t1, _ in gt_segs] + [t1 for _, t1, _, _ in pred_segs] + [1.0])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 2)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Pred", "GT"])
    ax.set_xlabel("Time (sec)")
    ax.set_title(f"GT vs Pred: {video_id}")

    bar_h = 0.6
    y_gt = 1.5 - bar_h / 2
    y_pr = 0.5 - bar_h / 2

    # ===== 关键修复：GT 不要用 black 写死；用 label2color[label] =====
    for t0, t1, lab in gt_segs:
        c = label2color.get(lab, (0, 0, 0, 1))
        ax.broken_barh([(t0, t1 - t0)], (y_gt, bar_h),
                       facecolors=c, edgecolors=c, alpha=0.9, linewidth=1.0)

    # Pred：同色系，但更透明一些，便于和 GT 区分
    for t0, t1, lab, score in pred_segs:
        c = label2color.get(lab, (0, 0, 0, 1))
        ax.broken_barh([(t0, t1 - t0)], (y_pr, bar_h),
                       facecolors=c, edgecolors=c, alpha=0.45, linewidth=0.8)

    # 图例（按 label）
    handles = [mpatches.Patch(color=label2color[lab], label=lab) for lab in all_labels]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


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
# plot_gt_pred_timeline_inline(
#     gt_path=gt_path,
#     pred_path=pred_path,
#     video_id="sbj_0",
#     subset="test",
#     score_thresh=0.01,
#     max_pred_per_label=80,
# )

# Opportunity
gt_path = "/home/lipei/project/WSDDN/test_results/Opportunity/pcl_0106/fold0/gt_for_anet.json"
pred_path = "/home/lipei/project/WSDDN/test_results/Opportunity/pcl_0106/fold0/predictions_test_full.json"

# RWHAR
# gt_path = "/home/lipei/project/WSDDN/test_results/RWHAR/wsddn_0105/fold0/gt_for_anet.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/RWHAR/wsddn_0105/fold0/predictions_test_full.json"

# SBHAR
# gt_path = "/home/lipei/project/WSDDN/test_results/SBHAR/wsddn_0105/fold0/gt_for_anet.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/SBHAR/wsddn_0105/fold0/predictions_test_full.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/SBHAR/wsddn_0105/fold0/predictions_test_window.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/SBHAR/wsddn_0103/fold0/predictions_test_full.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/SBHAR/wsddn_0103/fold0/predictions_test_window.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/SBHAR/wsddn_0103/fold0/predictions_test_full.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/SBHAR/wsddn_0103/fold0/predictions_test_window.json"

# WEAR
# gt_path = "/home/lipei/project/WSDDN/test_results/WEAR/wsddn_0105/fold0/gt_for_anet.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WEAR/wsddn_0105/fold0/predictions_test_full.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WEAR/wsddn_0105/fold0/predictions_test_window.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WEAR/oicr_0103/fold0/predictions_test_full.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WEAR/oicr_0103/fold0/predictions_test_window.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WEAR/pcl_0103/fold0/predictions_test_full.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WEAR/pcl_0103/fold0/predictions_test_window.json"

# WEAR
# gt_path = "/home/lipei/project/WSDDN/test_results/WETLAB/wsddn_0105/fold0/gt_for_anet.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WETLAB/wsddn_0105/fold0/predictions_test_full.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WETLAB/wsddn_0105/fold0/predictions_test_window.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WETLAB/oicr_0103/fold0/predictions_test_full.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WETLAB/oicr_0103/fold0/predictions_test_window.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WETALB/pcl_0103/fold0/predictions_test_full.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WETALB/pcl_0103/fold0/predictions_test_window.json"

plot_gt_pred_two_rows_inline(
    gt_path=gt_path,
    pred_path=pred_path,
    video_id="sbj_0",
    subset="test",
    # score_thresh=0.01,
    # topk_total=400,   # 预测多就调小点
)