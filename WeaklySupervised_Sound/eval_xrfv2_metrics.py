import os
import json
import math
import argparse
from collections import defaultdict
import numpy as np


# ============================================================
# Basic I/O
# ============================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_mean(values):
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if len(vals) else float("nan")


# ============================================================
# tIoU + AP/mAP (ActivityNet-style one-to-one matching)
# NOTE: unit doesn't matter as long as GT and Pred use same unit.
# Here segments are in time-step index (sample index).
# ============================================================
def tiou_1d(a, b):
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return 0.0 if union <= 0 else inter / union


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map(gt_by_vid, pred_by_vid, class_labels, tiou_thresholds):
    """
    Segment-level mAP:
      per-class, per-video one-to-one matching at each tIoU threshold.
    """
    gt_cv = defaultdict(list)     # (label, vid) -> list segs
    npos = defaultdict(int)       # label -> total GT instances
    for vid, gts in gt_by_vid.items():
        for g in gts:
            gt_cv[(g["label"], vid)].append(g["segment"])
            npos[g["label"]] += 1

    pred_c = defaultdict(list)    # label -> list (score, vid, seg)
    for vid, preds in pred_by_vid.items():
        for p in preds:
            pred_c[p["label"]].append((p["score"], vid, p["segment"]))

    mAP_per_t = []
    AP_per_t_per_class = {}

    for t in tiou_thresholds:
        ap_c = {}
        for lab in class_labels:
            n = npos.get(lab, 0)
            if n == 0:
                ap_c[lab] = float("nan")
                continue

            preds = sorted(pred_c.get(lab, []), key=lambda x: -x[0])
            if len(preds) == 0:
                ap_c[lab] = 0.0
                continue

            used = {}
            for (l2, vid), segs in gt_cv.items():
                if l2 == lab:
                    used[(lab, vid)] = np.zeros(len(segs), dtype=bool)

            tp = np.zeros(len(preds), dtype=float)
            fp = np.zeros(len(preds), dtype=float)

            for i, (_, vid, pseg) in enumerate(preds):
                gsegs = gt_cv.get((lab, vid), [])
                if not gsegs:
                    fp[i] = 1.0
                    continue
                ious = np.array([tiou_1d(pseg, gseg) for gseg in gsegs], dtype=float)
                j = int(np.argmax(ious))
                if ious[j] >= t and (not used[(lab, vid)][j]):
                    tp[i] = 1.0
                    used[(lab, vid)][j] = True
                else:
                    fp[i] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / float(n)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap_c[lab] = voc_ap(rec, prec)

        AP_per_t_per_class[t] = ap_c
        mAP_per_t.append(safe_mean(ap_c.values()))

    return mAP_per_t, AP_per_t_per_class


# ============================================================
# Load XRFV2 GT / Pred
# ============================================================
def load_xrfv2_gt(gt_path):
    """
    imu_annotations.json:
    {
      "database": {
        "0_1_3.h5": {
          "subset": "test",
          "annotations": [{"segment":[s,e],"label":"..."} , ...]
        },
        ...
      }
    }
    Segments are in time-step index (not seconds).
    """
    d = load_json(gt_path)
    db = d["database"]

    gt_by_vid = {}
    label_set = set()
    end_by_vid = {}

    for vid, info in db.items():
        anns = info.get("annotations", [])
        insts = []
        max_end = 0
        for ann in anns:
            lab = ann["label"]
            s, e = ann["segment"]
            s = int(s)
            e = int(e)
            if e > s:
                insts.append({"label": lab, "segment": (float(s), float(e))})
                label_set.add(lab)
                max_end = max(max_end, e)
        gt_by_vid[vid] = insts
        end_by_vid[vid] = int(max_end)

    # map labels to 0..C-1 (stable order)
    labels_sorted = sorted(list(label_set))
    label_to_id = {lab: i for i, lab in enumerate(labels_sorted)}

    return gt_by_vid, end_by_vid, label_to_id


def load_xrfv2_pred(pred_path, label_to_id, conf_thresh=None):
    """
    predictions_test_full/window.json:
    {
      "results": {
        "0_1_3.h5": [{"label": "...", "score": 0.49, "segment":[s,e]}, ...],
        ...
      }
    }
    Segments are in time-step index (float/int).
    We ignore labels not in GT label_to_id to keep label space consistent.
    """
    d = load_json(pred_path)
    res = d.get("results", {})
    out = {}
    for vid, items in res.items():
        insts = []
        for it in items:
            lab = it.get("label")
            seg = it.get("segment")
            sc = float(it.get("score", 0.0))
            if lab is None or seg is None:
                continue
            if lab not in label_to_id:
                continue
            if conf_thresh is not None and sc <= conf_thresh:
                continue
            s, e = seg
            s = float(s)
            e = float(e)
            if e > s:
                insts.append({"label": lab, "segment": (s, e), "score": sc})
        out[vid] = insts
    return out


# ============================================================
# Rasterize (segments -> per-time-step labels)
# Here unit already is time-step index, so we don't multiply by fps.
# ============================================================
def rasterize_video(T, gt_segs, pred_segs, label_to_id):
    """
    Build arrays:
      gt_arr[t] in {0..C}   (0=null, action = label_id+1)
      pr_arr[t] in {0..C}
    Overlap preds resolved by per-time-step max score.
    """
    if T <= 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    gt_arr = np.zeros((T,), dtype=np.int32)

    # GT fill
    for g in gt_segs:
        lab = g["label"]
        if lab not in label_to_id:
            continue
        cid = label_to_id[lab] + 1
        s, e = g["segment"]
        s_idx = max(0, int(math.floor(s)))
        e_idx = min(T, int(math.ceil(e)))
        if e_idx > s_idx:
            gt_arr[s_idx:e_idx] = cid

    # Pred fill (max score per time-step)
    pr_arr = np.zeros((T,), dtype=np.int32)
    best = np.full((T,), -1e30, dtype=float)
    for p in pred_segs:
        lab = p["label"]
        if lab not in label_to_id:
            continue
        cid = label_to_id[lab] + 1
        sc = float(p["score"])
        s, e = p["segment"]
        s_idx = max(0, int(math.floor(s)))
        e_idx = min(T, int(math.ceil(e)))
        if e_idx <= s_idx:
            continue
        sl = slice(s_idx, e_idx)
        mask = sc > best[sl]
        if np.any(mask):
            idxs = np.where(mask)[0] + s_idx
            best[idxs] = sc
            pr_arr[idxs] = cid

    return gt_arr, pr_arr


# ============================================================
# P/R/F1 on sample-level (macro over classes)
# ============================================================
def per_class_counts(gt_arr, pr_arr, labels):
    out = {}
    for c in labels:
        gt_c = (gt_arr == c)
        pr_c = (pr_arr == c)
        tp = int(np.sum(gt_c & pr_c))
        fp = int(np.sum((~gt_c) & pr_c))
        fn = int(np.sum(gt_c & (~pr_c)))
        tn = int(np.sum((~gt_c) & (~pr_c)))
        out[c] = (tp, tn, fp, fn)
    return out


def macro_prf1(gt_arr, pr_arr, labels):
    counts = per_class_counts(gt_arr, pr_arr, labels)
    ps, rs, f1s = [], [], []
    for c in labels:
        tp, tn, fp, fn = counts[c]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        ps.append(p); rs.append(r); f1s.append(f1)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(f1s))


# ============================================================
# UODIFM: EXACT pasted TAL implementation (do not change)
# ============================================================
def compute_misalignment_measures(gt, pred, classes, has_null=True):
    underfill_ratio, deletion_ratio, merge_ratio, fragmentation_ratio, insertion_ratio, overfill_ratio = \
        np.zeros(classes.stop), np.zeros(classes.stop), np.zeros(classes.stop), np.zeros(classes.stop), np.zeros(classes.stop), np.zeros(classes.stop)
    for class_label in classes:
        true_positive_count, true_negative_count, false_positive_count, false_negative_count = 0, 0, 0, 0
        curr_seg = []
        gt_pred_pairs = list(zip(gt, pred))
        i = 0
        while i < len(gt_pred_pairs):
            g_instance, p_instance = gt_pred_pairs[i]
            if (g_instance == class_label) and (p_instance == class_label):
                true_positive_count += 1
                curr_seg.append(i)
                i += 1
            elif (g_instance != class_label) and (p_instance != class_label):
                true_negative_count += 1
                curr_seg = []
                i += 1
            elif (g_instance == class_label) and (p_instance != class_label):
                try:
                    gt[i+1]
                except IndexError:
                    if (gt[i-1] == class_label):
                        underfill_ratio[class_label] += 1
                        false_negative_count += 1
                        i += 1
                    else:
                        deletion_ratio[class_label] += 1
                        false_negative_count += 1
                        i += 1
                    break
                if len(curr_seg) > 0:
                    j = i
                    length = 0
                    fragmentation = False
                    while gt[j] == class_label and j < len(gt_pred_pairs):
                        if pred[j] == class_label:
                            fragmentation_ratio[class_label] += length
                            false_negative_count += length
                            i = j
                            fragmentation = True
                            break
                        length += 1
                        j += 1
                        if j ==  len(gt_pred_pairs):
                            break
                    if not fragmentation:
                        underfill_ratio[class_label] += length
                        false_negative_count += length
                        i = j
                elif len(curr_seg) == 0:
                    j = i
                    length = 0
                    underfill = False
                    while gt[j] == class_label:
                        if pred[j] == class_label:
                            underfill_ratio[class_label] += length
                            false_negative_count += length
                            i = j
                            underfill = True
                            break
                        length += 1
                        j += 1
                        if j ==  len(gt_pred_pairs):
                            break
                    if not underfill:
                        deletion_ratio[class_label] += length
                        false_negative_count += length
                        i = j
            elif (g_instance != class_label) and (p_instance == class_label):
                try:
                    gt[i+1]
                except IndexError:
                    if (gt[i-1] == class_label):
                        overfill_ratio[class_label] += 1
                        false_positive_count += 1
                        i += 1
                    else:
                        insertion_ratio[class_label] += 1
                        false_positive_count += 1
                        i += 1
                    break
                if len(curr_seg) > 0:
                    j = i
                    length = 0
                    overfill = False
                    while gt[j] != class_label:
                        if pred[j] != class_label:
                            overfill_ratio[class_label] += length
                            false_positive_count += length
                            i = j
                            overfill = True
                            break
                        length += 1
                        j += 1
                        if j ==  len(gt_pred_pairs):
                            break
                    if not overfill:
                        merge_ratio[class_label] += length
                        false_positive_count += length
                        i = j
                elif len(curr_seg) == 0:
                    j = i
                    length = 0
                    insert = False
                    while gt[j] != class_label:
                        if pred[j] != class_label:
                            insertion_ratio[class_label] += length
                            false_positive_count += length
                            i = j
                            insert = True
                            break
                        length += 1
                        j += 1
                        if j ==  len(gt_pred_pairs):
                            break
                    if not insert:
                        overfill_ratio[class_label] += length
                        false_positive_count += length
                        i = j
        if (true_negative_count + false_negative_count) > 0:
            underfill_ratio[class_label] /= (true_negative_count + false_negative_count)
            deletion_ratio[class_label] /= (true_negative_count + false_negative_count)
            fragmentation_ratio[class_label] /= (true_negative_count + false_negative_count)
        if (true_positive_count + false_positive_count) > 0:
            merge_ratio[class_label] /= (true_positive_count + false_positive_count)
            overfill_ratio[class_label] /= (true_positive_count + false_positive_count)
            insertion_ratio[class_label] /= (true_positive_count + false_positive_count)
    if has_null:
        return underfill_ratio[1:], deletion_ratio[1:], fragmentation_ratio[1:], insertion_ratio[1:], overfill_ratio[1:], merge_ratio[1:]
    else:
        return underfill_ratio, deletion_ratio, fragmentation_ratio, insertion_ratio, overfill_ratio, merge_ratio


# ============================================================
# Evaluate XRFV2 (full + window)  [GT path fixed]
# ============================================================
FIXED_GT_PATH = "/home/lipei/XRFV2/imu_annotations.json"

def evaluate_xrfv2(
    pred_dir,
    gt_path=FIXED_GT_PATH,
    pred_full="predictions_test_full.json",
    pred_window="predictions_test_window.json",
    tiou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
    use_conf_thresh=False,
    conf_thresh=0.0,
    apply_conf_to_map=False,
    output_name="metrics_summary_xrfv2.json",
):
    pred_dir = os.path.abspath(pred_dir)

    if not os.path.exists(gt_path):
        raise FileNotFoundError(gt_path)

    gt_by_vid, end_by_vid, label_to_id = load_xrfv2_gt(gt_path)
    class_labels = sorted(label_to_id.keys())

    C = len(class_labels)
    num_with_null = C + 1
    labels_all = list(range(num_with_null))
    labels_nonnull = list(range(1, num_with_null))

    def eval_mode(pred_filename):
        pred_path = os.path.join(pred_dir, pred_filename)
        if not os.path.exists(pred_path):
            raise FileNotFoundError(pred_path)

        th_used = float(conf_thresh) if use_conf_thresh else None

        pred_raw = load_xrfv2_pred(pred_path, label_to_id, conf_thresh=None)
        vids = sorted([v for v in pred_raw.keys() if v in gt_by_vid])

        # mAP preds
        if apply_conf_to_map and use_conf_thresh:
            pred_map = load_xrfv2_pred(pred_path, label_to_id, conf_thresh=conf_thresh)
        else:
            pred_map = pred_raw

        gt_subset = {v: gt_by_vid.get(v, []) for v in vids}
        pred_subset_map = {v: pred_map.get(v, []) for v in vids}

        mAP_per_t, _ = evaluate_map(gt_subset, pred_subset_map, class_labels, tiou_thresholds)
        mAP_mean = float(np.mean(mAP_per_t))

        # sample-level preds
        if use_conf_thresh:
            pred_sample = load_xrfv2_pred(pred_path, label_to_id, conf_thresh=conf_thresh)
        else:
            pred_sample = pred_raw

        gt_cat_list, pr_cat_list = [], []
        for v in vids:
            T = int(end_by_vid.get(v, 0))
            if T <= 0:
                continue
            gt_arr, pr_arr = rasterize_video(
                T=T,
                gt_segs=gt_subset.get(v, []),
                pred_segs=pred_sample.get(v, []),
                label_to_id=label_to_id
            )
            if gt_arr.size == 0:
                continue
            gt_cat_list.append(gt_arr)
            pr_cat_list.append(pr_arr)

        if not gt_cat_list:
            P_all = R_all = F1_all = float("nan")
            P_nn = R_nn = F1_nn = float("nan")
            uod = {k: float("nan") for k in ["UR", "OR", "DR", "IR", "FR", "MR"]}
        else:
            gt_cat = np.concatenate(gt_cat_list, axis=0)
            pr_cat = np.concatenate(pr_cat_list, axis=0)

            P_all, R_all, F1_all = macro_prf1(gt_cat, pr_cat, labels_all)
            P_nn, R_nn, F1_nn = macro_prf1(gt_cat, pr_cat, labels_nonnull)

            UR, DR, FR, IR, OR, MR = compute_misalignment_measures(gt_cat, pr_cat, range(num_with_null), has_null=True)
            uod = {
                "UR": float(np.mean(UR)),
                "OR": float(np.mean(OR)),
                "DR": float(np.mean(DR)),
                "IR": float(np.mean(IR)),
                "FR": float(np.mean(FR)),
                "MR": float(np.mean(MR)),
            }

        return {
            "conf_thresh_used": th_used,
            "mAP_per_tiou": [float(x) for x in mAP_per_t],
            "mAP_mean": float(mAP_mean),

            "P_macro": float(P_all),
            "R_macro": float(R_all),
            "F1_macro": float(F1_all),

            "P_macro_nonnull": float(P_nn),
            "R_macro_nonnull": float(R_nn),
            "F1_macro_nonnull": float(F1_nn),

            "UODIFM": uod,
            "num_sequences": len(vids),
        }

    out = {
        "pred_dir": pred_dir,
        "gt_used": gt_path,
        "tiou_thresholds": list(map(float, tiou_thresholds)),
        "thresholding": {
            "use_conf_thresh": bool(use_conf_thresh),
            "conf_thresh": float(conf_thresh),
            "apply_conf_to_map": bool(apply_conf_to_map),
        },
        "modes": {
            "test_full": eval_mode(pred_full),
            "test_window": eval_mode(pred_window),
        }
    }

    out_path = os.path.join(pred_dir, output_name)
    dump_json(out, out_path)
    return out_path


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_dir",
        type=str,
        default="/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_2024/",
        help="Directory that contains predictions_test_full.json and predictions_test_window.json"
    )
    ap.add_argument("--tiou", type=str, default="0.3,0.4,0.5,0.6,0.7")

    ap.add_argument("--use_conf_thresh", action="store_true")
    ap.add_argument("--conf_thresh", type=float, default=0.0)
    ap.add_argument("--apply_conf_to_map", action="store_true")

    args = ap.parse_args()
    tiou_thresholds = tuple(float(x.strip()) for x in args.tiou.split(","))

    out_path = evaluate_xrfv2(
        pred_dir=args.pred_dir,
        gt_path=FIXED_GT_PATH,
        tiou_thresholds=tiou_thresholds,
        use_conf_thresh=args.use_conf_thresh,
        conf_thresh=args.conf_thresh,
        apply_conf_to_map=args.apply_conf_to_map,
    )
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()



# python eval_xrfv2_metrics.py --pred_dir /home/lipei/project/WSDDN/test_results/xrfv2/xxx/
# python eval_xrfv2_metrics.py --use_conf_thresh --conf_thresh 0.15
