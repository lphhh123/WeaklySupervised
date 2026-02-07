import os
import json
from typing import Dict, List, Tuple
import numpy as np


def load_gt_pred(gt_path: str, pred_path: str, subset: str = "test"):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_json = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_json = json.load(f)

    gt_db = gt_json.get("database", {})
    gt_by_vid: Dict[str, List[dict]] = {}
    total_gt = 0

    for vid, info in gt_db.items():
        if info.get("subset", subset) != subset:
            continue
        annos = info.get("annotations", [])
        lst = []
        for a in annos:
            lab = a.get("label", None)
            seg = a.get("segment", None)
            if lab is None or seg is None:
                continue
            s, e = float(seg[0]), float(seg[1])
            if e <= s:
                continue
            lst.append({"label": str(lab), "segment": [s, e]})
        gt_by_vid[vid] = lst
        total_gt += len(lst)

    pred_res = pred_json.get("results", {})
    pred_by_vid: Dict[str, List[dict]] = {}
    total_pred = 0

    for vid, dets in pred_res.items():
        lst = []
        for d in dets:
            lab = d.get("label", None)
            seg = d.get("segment", None)
            sc = d.get("score", None)
            if lab is None or seg is None or sc is None:
                continue
            s, e = float(seg[0]), float(seg[1])
            if e <= s:
                continue
            lst.append({"label": str(lab), "segment": [s, e], "score": float(sc)})
        lst.sort(key=lambda x: x["score"], reverse=True)
        pred_by_vid[vid] = lst
        total_pred += len(lst)

    return gt_by_vid, pred_by_vid, total_gt, total_pred


def build_class_names_from_gt(gt_by_vid: Dict[str, List[dict]]) -> List[str]:
    s = set()
    for gts in gt_by_vid.values():
        for g in gts:
            s.add(str(g["label"]))
    return sorted(list(s))


def tiou_1d(seg_a, seg_b, eps=1e-6) -> float:
    s1, e1 = float(seg_a[0]), float(seg_a[1])
    s2, e2 = float(seg_b[0]), float(seg_b[1])
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / (union + eps)


def prf_micro(gt_by_vid, pred_by_vid, total_gt: int, thr: float, score_thr: float = 0.0):
    TP = 0
    FP = 0
    matched = {}  # (vid, gi)->bool

    for vid, dets in pred_by_vid.items():
        gts = gt_by_vid.get(vid, [])
        if not gts:
            for d in dets:
                if d["score"] >= score_thr:
                    FP += 1
            continue

        for det in dets:
            if det["score"] < score_thr:
                continue

            best_gi = -1
            best_iou = -1.0
            for gi, gt in enumerate(gts):
                if matched.get((vid, gi), False):
                    continue
                if gt["label"] != det["label"]:
                    continue
                iou = tiou_1d(det["segment"], gt["segment"])
                if iou >= thr and iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_gi >= 0:
                matched[(vid, best_gi)] = True
                TP += 1
            else:
                FP += 1

    FN = total_gt - TP
    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "P": P, "R": R, "F1": F1}


def segments_to_bins_frame(segments: List[List[float]], L: int) -> np.ndarray:
    bins = np.zeros((L,), dtype=np.uint8)
    if L <= 0:
        return bins
    for s, e in segments:
        s = float(s); e = float(e)
        if e <= s:
            continue
        a = int(np.floor(s))
        b = int(np.ceil(e))
        a = max(0, min(a, L))
        b = max(0, min(b, L))
        if b > a:
            bins[a:b] = 1
    return bins


def events_from_bins(bins: np.ndarray) -> List[Tuple[int, int]]:
    ev = []
    i = 0
    L = len(bins)
    while i < L:
        if bins[i] == 1:
            j = i + 1
            while j < L and bins[j] == 1:
                j += 1
            ev.append((i, j))
            i = j
        else:
            i += 1
    return ev


def uodifm_for_class(gt_bins: np.ndarray, pred_bins: np.ndarray):
    L = len(gt_bins)
    P = int(gt_bins.sum())
    N = int(L - P)

    gt_events = events_from_bins(gt_bins)
    pred_events = events_from_bins(pred_bins)

    UR = DR = FR = 0
    for (gs, ge) in gt_events:
        sl = pred_bins[gs:ge]
        if sl.sum() == 0:
            DR += (ge - gs)
            continue

        idx = np.where(sl == 1)[0]
        first = gs + int(idx[0])
        last = gs + int(idx[-1])

        UR += (first - gs) + (ge - (last + 1))

        if last > first:
            FR += int((pred_bins[first:last + 1] == 0).sum())

    OR = IR = MR = 0
    for (ps, pe) in pred_events:
        if gt_bins[ps:pe].sum() == 0:
            IR += (pe - ps)
            continue

        overlapped = []
        for (gs, ge) in gt_events:
            if max(ps, gs) < min(pe, ge):
                overlapped.append((gs, ge))

        neg_cnt = int((gt_bins[ps:pe] == 0).sum())

        if len(overlapped) <= 1:
            OR += neg_cnt
        else:
            first_gt = min(x[0] for x in overlapped)
            last_gt = max(x[1] for x in overlapped)
            inside_s = max(ps, first_gt)
            inside_e = min(pe, last_gt)
            mr = int((gt_bins[inside_s:inside_e] == 0).sum())
            MR += mr
            OR += (neg_cnt - mr)

    return {"UR": UR, "OR": OR, "DR": DR, "IR": IR, "FR": FR, "MR": MR, "P": P, "N": N}


def compute_uodifm(gt_by_vid, pred_by_vid, class_names: List[str], score_thr: float = 0.0):
    vids = sorted(set(gt_by_vid.keys()) | set(pred_by_vid.keys()))
    per_cls = {}

    for cls in class_names:
        acc = {"UR": 0, "OR": 0, "DR": 0, "IR": 0, "FR": 0, "MR": 0, "P": 0, "N": 0}

        for vid in vids:
            gts = [g["segment"] for g in gt_by_vid.get(vid, []) if g["label"] == cls]
            prs = [p["segment"] for p in pred_by_vid.get(vid, [])
                   if p["label"] == cls and float(p.get("score", 1.0)) >= score_thr]

            max_end = 0.0
            for s, e in gts:
                max_end = max(max_end, float(e))
            for s, e in prs:
                max_end = max(max_end, float(e))
            if max_end <= 0:
                continue

            L = int(np.ceil(max_end))
            if L <= 0:
                continue

            gt_bins = segments_to_bins_frame(gts, L)
            pred_bins = segments_to_bins_frame(prs, L)

            d = uodifm_for_class(gt_bins, pred_bins)
            for k in acc:
                acc[k] += d[k]

        P = acc["P"]
        N = acc["N"]

        per_cls[cls] = {
            "UR": acc["UR"] / P if P > 0 else 0.0,
            "DR": acc["DR"] / P if P > 0 else 0.0,
            "FR": acc["FR"] / P if P > 0 else 0.0,
            "OR": acc["OR"] / N if N > 0 else 0.0,
            "IR": acc["IR"] / N if N > 0 else 0.0,
            "MR": acc["MR"] / N if N > 0 else 0.0,

            "UR_len": acc["UR"], "OR_len": acc["OR"], "DR_len": acc["DR"],
            "IR_len": acc["IR"], "FR_len": acc["FR"], "MR_len": acc["MR"],
            "P_len": P, "N_len": N,
        }

    vals = [per_cls[c] for c in class_names if c in per_cls]
    macro = {
        "UR": float(np.mean([v["UR"] for v in vals])) if vals else 0.0,
        "OR": float(np.mean([v["OR"] for v in vals])) if vals else 0.0,
        "DR": float(np.mean([v["DR"] for v in vals])) if vals else 0.0,
        "IR": float(np.mean([v["IR"] for v in vals])) if vals else 0.0,
        "FR": float(np.mean([v["FR"] for v in vals])) if vals else 0.0,
        "MR": float(np.mean([v["MR"] for v in vals])) if vals else 0.0,
    }
    return per_cls, macro


def evaluate_prf_and_uodifm(
    gt_path: str,
    pred_path: str,
    out_dir: str,
    subset: str = "test",
    tious=(0.3, 0.4, 0.5, 0.6, 0.7),
    score_thr: float = 0.0,
):
    os.makedirs(out_dir, exist_ok=True)

    gt_by_vid, pred_by_vid, total_gt, total_pred = load_gt_pred(gt_path, pred_path, subset=subset)
    class_names = build_class_names_from_gt(gt_by_vid)

    # PRF micro
    prf_rows = []
    for thr in tious:
        d = prf_micro(gt_by_vid, pred_by_vid, total_gt, float(thr), score_thr=score_thr)
        prf_rows.append((float(thr), d))

    # Avg(PRF) over tIoU thresholds (mean of P/R/F1 across thresholds)
    avg_P = float(np.mean([d["P"] for _, d in prf_rows])) if prf_rows else 0.0
    avg_R = float(np.mean([d["R"] for _, d in prf_rows])) if prf_rows else 0.0
    avg_F1 = float(np.mean([d["F1"] for _, d in prf_rows])) if prf_rows else 0.0


    # UODIFM
    per_cls_u, macro_u = compute_uodifm(gt_by_vid, pred_by_vid, class_names, score_thr=score_thr)

    # save CSV
    prf_csv = os.path.join(out_dir, "prf_micro.csv")
    with open(prf_csv, "w", encoding="utf-8") as f:
        f.write("tiou,TP,FP,FN,P,R,F1\n")
        for thr, d in prf_rows:
            f.write(f"{thr:.2f},{d['TP']},{d['FP']},{d['FN']},{d['P']:.6f},{d['R']:.6f},{d['F1']:.6f}\n")
        # avg row
        f.write(f"avg,-,-,-,{avg_P:.6f},{avg_R:.6f},{avg_F1:.6f}\n")

    u_csv = os.path.join(out_dir, "uodifm_per_class.csv")
    with open(u_csv, "w", encoding="utf-8") as f:
        f.write("action,UR,OR,DR,IR,FR,MR,UR_len,OR_len,DR_len,IR_len,FR_len,MR_len,P_len,N_len\n")
        for cls in class_names:
            d = per_cls_u.get(cls)
            if d is None:
                continue
            f.write(
                f"{cls},{d['UR']:.6f},{d['OR']:.6f},{d['DR']:.6f},{d['IR']:.6f},{d['FR']:.6f},{d['MR']:.6f},"
                f"{d['UR_len']},{d['OR_len']},{d['DR_len']},{d['IR_len']},{d['FR_len']},{d['MR_len']},{d['P_len']},{d['N_len']}\n"
            )

    # report
    report_path = os.path.join(out_dir, "report.txt")
    lines = []
    lines.append("=" * 70)
    lines.append(f"GT   : {gt_path}")
    lines.append(f"Pred : {pred_path}")
    lines.append(f"Subset: {subset}")
    lines.append(f"Total GT segments = {total_gt}")
    lines.append(f"Total Pred segments = {total_pred}")
    lines.append(f"Classes = {len(class_names)}")
    lines.append(f"Score threshold used = {score_thr:.4f}")
    lines.append("-" * 70)
    lines.append("Micro PRF @ tIoU")
    for thr, d in prf_rows:
        lines.append(
            f"tIoU={thr:.2f} -> P={d['P']:.4f}, R={d['R']:.4f}, F1={d['F1']:.4f} "
            f"(TP={d['TP']}, FP={d['FP']}, FN={d['FN']})"
        )
        lines.append(
            f"Avg(PRF over tIoU={tious[0]:.2f}~{tious[-1]:.2f}) -> P={avg_P:.4f}, R={avg_R:.4f}, F1={avg_F1:.4f}"
        )
    lines.append("-" * 70)
    lines.append("UODIFM (macro over classes)")
    lines.append(
        f"UR={macro_u['UR']:.4f}, OR={macro_u['OR']:.4f}, DR={macro_u['DR']:.4f}, "
        f"IR={macro_u['IR']:.4f}, FR={macro_u['FR']:.4f}, MR={macro_u['MR']:.4f}"
    )
    lines.append("-" * 70)
    lines.append(f"Saved: {prf_csv}")
    lines.append(f"Saved: {u_csv}")
    lines.append("=" * 70)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    return {
        "prf_csv": prf_csv,
        "uodifm_csv": u_csv,
        "report": report_path,
        "macro_uodifm": macro_u,
        "class_names": class_names,
    }


# class_names = ["Stretching","Pouring Water","Writing","Cutting Fruit","Eating Fruit","Taking Medicine",
#         "Drinking Water","Sitting Down","Turning On/Off Eye Protection Lamp","Opening/Closing Curtains",
#         "Opening/Closing Windows","Typing","Opening Envelope","Throwing Garbage","Picking Fruit",
#         "Picking Up Items","Answering Phone","Using Mouse","Wiping Table","Writing on Blackboard",
#         "Washing Hands","Using Phone","Reading","Watering Plants","Walking","Getting Out of Bed",
#         "Standing Up","Lying Down","Standing Still","Lying Still"]


if __name__ == "__main__":
    gt_path = "/home/lipei/XRFV2/imu_annotations.json"
    pred_path = "/home/lipei/project/WSDDN/test_results/xrfv2/xrfv2_cnn_oicr_0112/predictions_test_full.json"
    out_dir = "/home/lipei/project/WSDDN/test_results/xrfv2/xrfv2_cnn_oicr_0112/eval_out_full"

    # 过滤低分预测
    score_thr = 0.0

    evaluate_prf_and_uodifm(
        gt_path=gt_path,
        pred_path=pred_path,
        out_dir=out_dir,
        subset="test",
        tious=(0.3, 0.4, 0.5, 0.6, 0.7),
        score_thr=score_thr,
    )
