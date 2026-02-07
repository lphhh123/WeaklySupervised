import os
import json
import glob
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
    Returns:
      mAP_per_t: list[float] len=T
      AP_per_t_per_class: dict[t][label]=AP
    """
    # GT grouped by (class, video)
    gt_cv = defaultdict(list)     # (label, vid) -> list segs
    npos = defaultdict(int)       # label -> total GT instances
    for vid, gts in gt_by_vid.items():
        for g in gts:
            gt_cv[(g["label"], vid)].append(g["segment"])
            npos[g["label"]] += 1

    # Pred grouped by class
    pred_c = defaultdict(list)    # label -> list (score, vid, seg)
    for vid, preds in pred_by_vid.items():
        for p in preds:
            pred_c[p["label"]].append((p["score"], vid, p["segment"]))

    AP_per_t_per_class = {}
    mAP_per_t = []

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
# Load GT / Pred
# ============================================================
def load_gt_for_anet(gt_path):
    """
    gt_for_anet.json:
      database[vid]: {duration, fps, annotations:[{label, segment:[s,e], label_id}, ...]}
    We return:
      gt_by_vid[vid] = list of {label, segment:(s,e)}
      fps_by_vid[vid], dur_by_vid[vid]
      label_to_id[label] = label_id
    """
    gt = load_json(gt_path)
    db = gt["database"]

    # label_to_id from label_id field
    label_to_id = {}
    for vid, info in db.items():
        for ann in info.get("annotations", []):
            lab = ann.get("label")
            lid = ann.get("label_id", None)
            if lab is not None and lid is not None and lab not in label_to_id:
                label_to_id[lab] = int(lid)

    gt_by_vid = {}
    fps_by_vid = {}
    dur_by_vid = {}

    for vid, info in db.items():
        fps_by_vid[vid] = float(info.get("fps", 0.0))
        dur_by_vid[vid] = float(info.get("duration", 0.0))
        insts = []
        for ann in info.get("annotations", []):
            lab = ann["label"]
            s, e = ann["segment"]
            s, e = float(s), float(e)
            if e > s:
                insts.append({"label": lab, "segment": (s, e)})
        gt_by_vid[vid] = insts

    return gt_by_vid, fps_by_vid, dur_by_vid, label_to_id


def load_predictions(pred_path, conf_thresh=None):
    """
    predictions_test_full/window.json:
      results[vid] = [{label, score, segment:[s,e]}, ...]
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
            if conf_thresh is not None and sc <= conf_thresh:
                continue
            s, e = float(seg[0]), float(seg[1])
            if e > s:
                insts.append({"label": lab, "segment": (s, e), "score": sc})
        out[vid] = insts
    return out


def read_conf_thresh(stats_path):
    """
    inference_stats_*.json includes "conf_thresh".
    """
    if not os.path.exists(stats_path):
        return None
    try:
        d = load_json(stats_path)
    except Exception:
        return None
    v = d.get("conf_thresh", None)
    return float(v) if isinstance(v, (int, float)) else None


# ============================================================
# Rasterize segments -> per-time-step labels (sample-level)
# (This mimics bock2024temporal pipeline: segment -> sequence)
# ============================================================
def rasterize_video(duration_sec, fps, gt_segs, pred_segs, label_to_id):
    """
    Build arrays:
      gt_arr[t] in {0..C}   (0=null, action = label_id+1)
      pr_arr[t] in {0..C}
    Resolve overlapping predictions by per-time-step max score winner.
    """
    if fps <= 0 or duration_sec <= 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    T = int(math.ceil(duration_sec * fps))
    gt_arr = np.zeros((T,), dtype=np.int32)

    # GT fill
    for g in gt_segs:
        lab = g["label"]
        if lab not in label_to_id:
            continue
        cid = label_to_id[lab] + 1
        s, e = g["segment"]
        s_idx = max(0, int(math.floor(s * fps)))
        e_idx = min(T, int(math.ceil(e * fps)))
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
        s_idx = max(0, int(math.floor(s * fps)))
        e_idx = min(T, int(math.ceil(e * fps)))
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
                # true positive
                true_positive_count += 1
                curr_seg.append(i)
                i += 1
            elif (g_instance != class_label) and (p_instance != class_label):
                # true negative
                true_negative_count += 1
                curr_seg = []
                i += 1
            elif (g_instance == class_label) and (p_instance != class_label):
                # false negative
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
                    # fragmenting or underfill
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
                    # deletion or underfill
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
                # false positive
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
                    # overfill or merge
                    j = i
                    length = 0
                    overfill = False
                    while gt[j] != class_label:
                        if pred[j] != class_label:
                            # overfill
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
                    # overfill or insert
                    j = i
                    length = 0
                    insert = False
                    while gt[j] != class_label:
                        if pred[j] != class_label:
                            # overfill
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
# Evaluate one experiment dir
# ============================================================
def evaluate_experiment_dir(
    exp_dir,
    tiou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
    use_conf_thresh=False,
    apply_conf_to_map=False,
    conf_thresh_override=None,
    gt_path_override=None,
):
    """
    exp_dir/
      fold0/ fold1/ ... foldK/
        predictions_test_full.json
        predictions_test_window.json
        inference_stats_test_full.json
        inference_stats_test_window.json
        gt_for_anet.json   (identical across folds)
    Output:
      exp_dir/metrics_summary_all_folds.json
    """
    exp_dir = os.path.abspath(exp_dir)
    fold_dirs = sorted([p for p in glob.glob(os.path.join(exp_dir, "fold*")) if os.path.isdir(p)])
    if not fold_dirs:
        raise RuntimeError(f"No fold* directories found: {exp_dir}")

    # use fold0 gt as reference (as requested)
    gt_path = os.path.join(exp_dir, "fold0", "gt_for_anet.json")
    if not os.path.exists(gt_path):
        # fallback to first fold
        gt_path = os.path.join(fold_dirs[0], "gt_for_anet.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"gt_for_anet.json not found in fold0 or first fold: {exp_dir}")

    gt_by_vid_all, fps_by_vid_all, dur_by_vid_all, label_to_id = load_gt_for_anet(gt_path)
    class_labels = sorted(label_to_id.keys())
    C = max(label_to_id.values()) + 1
    num_with_null = C + 1
    labels_all = list(range(num_with_null))      # include null
    labels_nonnull = list(range(1, num_with_null))

    def eval_mode(mode_name):
        """
        mode_name: 'test_full' or 'test_window'
        """
        per_fold = []

        for fd in fold_dirs:
            pred_path = os.path.join(fd, f"predictions_{mode_name}.json")
            if not os.path.exists(pred_path):
                raise FileNotFoundError(pred_path)

            # threshold (optional)
            th_src = "none"
            th = None
            if use_conf_thresh:
                if conf_thresh_override is not None:
                    th = float(conf_thresh_override)
                    th_src = "override"
                else:
                    stats_path = os.path.join(fd, f"inference_stats_{mode_name}.json")
                    th_stats = read_conf_thresh(stats_path)
                    if th_stats is not None:
                        th = float(th_stats)
                        th_src = "inference_stats"
                    else:
                        th = 0.0
                        th_src = "fallback_0.0"

            # load preds (for selecting eval vids)
            pred_raw = load_predictions(pred_path, conf_thresh=None)
            eval_vids = sorted(list(pred_raw.keys()))

            # build per-fold GT subset (only those vids)
            gt_by_vid = {vid: gt_by_vid_all.get(vid, []) for vid in eval_vids}
            fps_by_vid = {vid: fps_by_vid_all.get(vid, 0.0) for vid in eval_vids}
            dur_by_vid = {vid: dur_by_vid_all.get(vid, 0.0) for vid in eval_vids}

            # mAP predictions: optionally thresholded (to mimic bock2024temporal)
            if apply_conf_to_map and use_conf_thresh:
                pred_map = load_predictions(pred_path, conf_thresh=th)
                th_used_map = th
            else:
                pred_map = pred_raw
                th_used_map = None

            mAP_per_t, _ = evaluate_map(gt_by_vid, pred_map, class_labels, tiou_thresholds)
            mAP_mean = float(np.mean(mAP_per_t))

            # sample-level predictions (thresholded only if enabled)
            if use_conf_thresh:
                pred_sample = load_predictions(pred_path, conf_thresh=th)
                th_used_sample = th
            else:
                pred_sample = pred_raw
                th_used_sample = None

            # rasterize + concat (TAL code concatenates sequences; we follow that)
            gt_cat_list = []
            pr_cat_list = []
            for vid in eval_vids:
                fps = fps_by_vid.get(vid, 0.0)
                dur = dur_by_vid.get(vid, 0.0)
                gt_arr, pr_arr = rasterize_video(
                    dur, fps,
                    gt_by_vid.get(vid, []),
                    pred_sample.get(vid, []),
                    label_to_id
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

                # P/R/F1 (macro): report both incl-null and non-null
                P_all, R_all, F1_all = macro_prf1(gt_cat, pr_cat, labels_all)
                P_nn, R_nn, F1_nn = macro_prf1(gt_cat, pr_cat, labels_nonnull)

                # UODIFM (exact impl): returns arrays excluding null
                UR, DR, FR, IR, OR, MR = compute_misalignment_measures(gt_cat, pr_cat, range(num_with_null), has_null=True)
                uod = {
                    "UR": float(np.mean(UR)),
                    "OR": float(np.mean(OR)),
                    "DR": float(np.mean(DR)),
                    "IR": float(np.mean(IR)),
                    "FR": float(np.mean(FR)),
                    "MR": float(np.mean(MR)),
                }

            per_fold.append({
                "fold": os.path.basename(fd),
                "conf_thresh": th,
                "conf_thresh_source": th_src,
                "conf_thresh_used_for_sample": th_used_sample,
                "conf_thresh_used_for_map": th_used_map,

                "mAP_per_tiou": [float(x) for x in mAP_per_t],
                "mAP_mean": float(mAP_mean),

                "P_macro": float(P_all),
                "R_macro": float(R_all),
                "F1_macro": float(F1_all),

                "P_macro_nonnull": float(P_nn),
                "R_macro_nonnull": float(R_nn),
                "F1_macro_nonnull": float(F1_nn),

                "UODIFM": uod
            })

        # mean over folds (NO std)
        mean_block = {
            "mAP_per_tiou": [safe_mean([r["mAP_per_tiou"][i] for r in per_fold]) for i in range(len(tiou_thresholds))],
            "mAP_mean": safe_mean([r["mAP_mean"] for r in per_fold]),

            "P_macro": safe_mean([r["P_macro"] for r in per_fold]),
            "R_macro": safe_mean([r["R_macro"] for r in per_fold]),
            "F1_macro": safe_mean([r["F1_macro"] for r in per_fold]),

            "P_macro_nonnull": safe_mean([r["P_macro_nonnull"] for r in per_fold]),
            "R_macro_nonnull": safe_mean([r["R_macro_nonnull"] for r in per_fold]),
            "F1_macro_nonnull": safe_mean([r["F1_macro_nonnull"] for r in per_fold]),

            "UODIFM": {
                "UR": safe_mean([r["UODIFM"]["UR"] for r in per_fold]),
                "OR": safe_mean([r["UODIFM"]["OR"] for r in per_fold]),
                "DR": safe_mean([r["UODIFM"]["DR"] for r in per_fold]),
                "IR": safe_mean([r["UODIFM"]["IR"] for r in per_fold]),
                "FR": safe_mean([r["UODIFM"]["FR"] for r in per_fold]),
                "MR": safe_mean([r["UODIFM"]["MR"] for r in per_fold]),
            }
        }

        return {"per_fold": per_fold, "mean_over_folds": mean_block}

    out = {
        "experiment_dir": exp_dir,
        "gt_used": gt_path,
        "tiou_thresholds": list(map(float, tiou_thresholds)),

        "thresholding": {
            "use_conf_thresh": bool(use_conf_thresh),
            "apply_conf_to_map": bool(apply_conf_to_map),
            "conf_thresh_override": conf_thresh_override,
            "note": "If use_conf_thresh=True, we filter predictions by conf_thresh before sample-level metrics; "
                    "if apply_conf_to_map=True, the same filtering is also applied before mAP (as in bock2024temporal pipeline)."
        },

        "modes": {
            "test_full": eval_mode("test_full"),
            "test_window": eval_mode("test_window"),
        },

        "notes": {
            "mAP": "segment-level ActivityNet-style AP/mAP under tIoU thresholds",
            "P/R/F1": "sample(time-step)-level after rasterizing segments to label sequences (reported as macro averages)",
            "UODIFM": "computed on rasterized sequences using the exact compute_misalignment_measures implementation (Ward-style categories)"
        }
    }

    out_path = os.path.join(exp_dir, "metrics_summary_all_folds.json")
    dump_json(out, out_path)
    return out_path


# ============================================================
# Evaluate many experiments in one call
# ============================================================
def run_all_experiments(exp_dirs, **kwargs):
    saved = []
    for d in exp_dirs:
        out_path = evaluate_experiment_dir(d, **kwargs)
        saved.append(out_path)
        print(f"[OK] {d} -> {out_path}")
    return saved


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dirs", nargs="+", required=True,
                    help="List of experiment directories (each contains fold*/).")
    ap.add_argument("--tiou", type=str, default="0.3,0.4,0.5,0.6,0.7")

    # thresholding options
    ap.add_argument("--use_conf_thresh", action="store_true",
                    help="If set, filter predictions by each fold's inference_stats_*.json conf_thresh before sample-level metrics.")
    ap.add_argument("--apply_conf_to_map", action="store_true",
                    help="If set, apply the same conf_thresh filtering before mAP as well (bock2024temporal-style).")
    ap.add_argument("--conf_thresh_override", type=float, default=None,
                    help="Override conf_thresh for all folds/modes (if provided).")

    args = ap.parse_args()
    tiou_thresholds = [float(x.strip()) for x in args.tiou.split(",")]

    run_all_experiments(
        args.exp_dirs,
        tiou_thresholds=tiou_thresholds,
        use_conf_thresh=args.use_conf_thresh,
        apply_conf_to_map=args.apply_conf_to_map,
        conf_thresh_override=args.conf_thresh_override,
    )


if __name__ == "__main__":
    main()


#
# python eval_folds_metrics.py --exp_dirs wsddn_0105 pcl_0104
#
#
# python eval_folds_metrics.py --exp_dirs wsddn_0105 pcl_0104 --use_conf_thresh
#
#
# python eval_folds_metrics.py --exp_dirs wsddn_0105 pcl_0104 --use_conf_thresh --apply_conf_to_map
#
#
# python eval_folds_metrics.py --exp_dirs wsddn_0105 pcl_0104 --use_conf_thresh --conf_thresh_override 0.15


# metrics_summary_all_folds.json
