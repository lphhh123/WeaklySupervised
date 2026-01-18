import os
import json
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def evaluate_extra_metrics_and_plots(
    gt_path: str,
    pred_path: str,
    class_names: list,
    tious,
    result_dir: str,
    test_mode: str,
    subset_name: str = "test",
    confusion_tiou: float = 0.5,
    include_bg: bool = True,
    # ===== UR/OR/DR/IR/FR/MR 的时间单位选择 =====
    # "frame"：按帧/index（每个bin=1帧） —— 可以根据pred和GT的json直接读取
    # "sec"  ：按秒（每个bin=1秒） —— 需要 seconds_per_unit
    uodifm_unit: str = "frame",
    seconds_per_unit: Optional[float] = None,  # 仅当 uodifm_unit="sec" 才需要；若 segment 本身是秒就填1.0
    # ===== 绘图参数 =====
    cm_value_thresh: float = -0.0,           # 混淆矩阵标注阈值（归一化值 >= 该值才写文字）
):
    """
    产物（保存在 result_dir）：
      1) per_action_prf_{test_mode}.csv
      2) per_action_uodifm_{test_mode}.csv
      3) confusion_matrix_{test_mode}_tiou{thr}.png
    返回：
      report_lines: list[str] —— 可直接 report_content.extend(report_lines)
      artifacts: dict —— 各文件路径
    """
    assert uodifm_unit in ["frame", "sec"], f"uodifm_unit must be 'frame' or 'sec', got {uodifm_unit}"
    tious = [float(x) for x in tious]
    os.makedirs(result_dir, exist_ok=True)

    # -------------------------
    # basic helpers
    # -------------------------
    def _tiou_1d(seg_a, seg_b, eps=1e-6):
        s1, e1 = float(seg_a[0]), float(seg_a[1])
        s2, e2 = float(seg_b[0]), float(seg_b[1])
        inter = max(0.0, min(e1, e2) - max(s1, s2))
        union = (e1 - s1) + (e2 - s2) - inter
        return inter / (union + eps)

    def _load_gt_pred(gt_path, pred_path, subset_name="test"):
        """
        GT: {"database":{vid:{"subset":"test","annotations":[{"label":str,"segment":[s,e]}, ...]}}}
        Pred: {"results":{vid:[{"label":str,"segment":[s,e],"score":float}, ...]}}
        """
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_json = json.load(f)

        gt_db = gt_json.get("database", {})
        gt_by_vid = {}
        total_gt = 0
        max_end_gt = 0.0

        for vid, info in gt_db.items():
            if info.get("subset", subset_name) != subset_name:
                continue
            annos = info.get("annotations", [])
            lst = []
            for a in annos:
                label = a.get("label", None)
                seg = a.get("segment", None)
                if label is None or seg is None:
                    continue
                s, e = float(seg[0]), float(seg[1])
                lst.append({"label": str(label), "segment": [s, e]})
                max_end_gt = max(max_end_gt, e)
            gt_by_vid[vid] = lst
            total_gt += len(lst)

        pred_res = pred_json.get("results", {})
        pred_by_vid = {}
        total_pred = 0
        max_end_pr = 0.0

        for vid, dets in pred_res.items():
            lst = []
            for d in dets:
                label = d.get("label", None)
                seg = d.get("segment", None)
                score = d.get("score", None)
                if label is None or seg is None or score is None:
                    continue
                s, e = float(seg[0]), float(seg[1])
                lst.append({"label": str(label), "segment": [s, e], "score": float(score)})
                max_end_pr = max(max_end_pr, e)
            lst.sort(key=lambda x: x["score"], reverse=True)
            pred_by_vid[vid] = lst
            total_pred += len(lst)

        return gt_by_vid, pred_by_vid, total_gt, total_pred, max_end_gt, max_end_pr

    # -------------------------
    # PRF (micro) overall
    # -------------------------
    def _overall_prf(gt_by_vid, pred_by_vid, total_gt, thr):
        TP = 0
        FP = 0
        matched = {}  # (vid, gi)->bool (同阈值下每个GT最多匹配一次)

        for vid, dets in pred_by_vid.items():
            gts = gt_by_vid.get(vid, [])
            if not gts:
                FP += len(dets)
                continue

            for det in dets:
                best_gi = -1
                best_iou = -1.0
                for gi, gt in enumerate(gts):
                    if matched.get((vid, gi), False):
                        continue
                    if gt["label"] != det["label"]:
                        continue
                    iou = _tiou_1d(det["segment"], gt["segment"])
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

    # -------------------------
    # PRF per action
    # -------------------------
    def _per_action_prf(gt_by_vid, pred_by_vid, cls, thr):
        TP = FP = FN = 0
        for vid, gts in gt_by_vid.items():
            gt_list = [g["segment"] for g in gts if g["label"] == cls]
            dets = pred_by_vid.get(vid, [])
            pred_list = [d for d in dets if d["label"] == cls]  # dets已按score降序

            matched_gt = [False] * len(gt_list)

            for det in pred_list:
                best_i = -1
                best_iou = -1.0
                for gi, seg_g in enumerate(gt_list):
                    if matched_gt[gi]:
                        continue
                    iou = _tiou_1d(det["segment"], seg_g)
                    if iou >= thr and iou > best_iou:
                        best_iou = iou
                        best_i = gi

                if best_i >= 0:
                    matched_gt[best_i] = True
                    TP += 1
                else:
                    FP += 1

            FN += sum(1 for x in matched_gt if not x)

        P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
        return {"TP": TP, "FP": FP, "FN": FN, "P": P, "R": R, "F1": F1}

    # -------------------------
    # confusion matrix (detection-style with BG)
    # -------------------------
    def _confusion_matrix(gt_by_vid, pred_by_vid, labels, thr, include_bg=True):
        """
        Frame-wise / bin-wise confusion matrix:
          - row = GT label at each time bin
          - col = Pred label at each time bin
        Notes:
          * 'thr' is NOT used here (kept only for compatibility and naming).
          * Pred label at a bin is chosen as the highest-score detection covering that bin.
          * If a bin is uncovered by any GT/Pred segment, it is BG (when include_bg=True).
        """
        BG = "BG"
        labels2 = labels + ([BG] if include_bg else [])
        label2id = {lab: i for i, lab in enumerate(labels2)}
        M = np.zeros((len(labels2), len(labels2)), dtype=np.int64)  # row=GT col=Pred

        # ---- helper: map continuous seg [s,e] -> discrete [a,b) bins ----
        def _seg_to_ab(s, e, L):
            s = float(s);
            e = float(e)
            if e <= s or L <= 0:
                return None
            if uodifm_unit == "frame":
                a = int(np.floor(s))
                b = int(np.ceil(e))
            else:
                # uodifm_unit == "sec"
                if seconds_per_unit is None:
                    raise ValueError("Frame-wise CM with uodifm_unit='sec' requires seconds_per_unit")
                sp = float(seconds_per_unit)
                a = int(np.floor(s * sp))
                b = int(np.ceil(e * sp))

            # clip
            a = max(0, min(a, L))
            b = max(0, min(b, L))
            if b <= a:
                return None
            return a, b

        vids = sorted(set(gt_by_vid.keys()) | set(pred_by_vid.keys()))

        for vid in vids:
            gts = gt_by_vid.get(vid, [])
            dets = pred_by_vid.get(vid, [])

            # ---- determine timeline length L ----
            max_end = 0.0
            for g in gts:
                max_end = max(max_end, float(g["segment"][1]))
            for d in dets:
                max_end = max(max_end, float(d["segment"][1]))

            if max_end <= 0:
                continue

            if uodifm_unit == "frame":
                L = int(np.ceil(max_end))
            else:
                # sec bins
                if seconds_per_unit is None:
                    raise ValueError("uodifm_unit='sec' requires seconds_per_unit")
                L = int(np.ceil(max_end * float(seconds_per_unit)))

            if L <= 0:
                continue

            # ---- build gt label per bin ----
            if include_bg:
                gt_lab = np.full((L,), BG, dtype=object)
            else:
                # If not including BG, we will skip bins that map to BG
                gt_lab = np.full((L,), None, dtype=object)

            # GT通常不重叠；如有重叠，这里“后写覆盖前写”
            for g in gts:
                lab = str(g["label"])
                seg = g["segment"]
                ab = _seg_to_ab(seg[0], seg[1], L)
                if ab is None:
                    continue
                a, b = ab
                gt_lab[a:b] = lab

            # ---- build pred label per bin by highest score ----
            if include_bg:
                pred_lab = np.full((L,), BG, dtype=object)
            else:
                pred_lab = np.full((L,), None, dtype=object)

            pred_score = np.full((L,), -1e30, dtype=np.float32)

            # 可选：设置一个分数阈值，过滤低分预测
            score_thr = None  # e.g. 0.1 / 0.2 / 0.3
            for d in dets:
                lab = str(d["label"])
                seg = d["segment"]
                sc = float(d["score"])
                if score_thr is not None and sc < score_thr:
                    continue
                ab = _seg_to_ab(seg[0], seg[1], L)
                if ab is None:
                    continue
                a, b = ab

                # 对每个bin选择覆盖它的最高score类别
                sl = pred_score[a:b]
                mask = sc > sl
                if np.any(mask):
                    pred_score[a:b][mask] = sc
                    pred_lab[a:b][mask] = lab

            # ---- accumulate confusion ----
            for t in range(L):
                gl = gt_lab[t]
                pl = pred_lab[t]

                if not include_bg:
                    # 没有BG时：只统计 gl/pl 都在 labels 内的bin
                    if gl is None or pl is None:
                        continue

                if include_bg:
                    gl = gl if gl in label2id else BG
                    pl = pl if pl in label2id else BG
                else:
                    if gl not in label2id or pl not in label2id:
                        continue

                M[label2id[gl], label2id[pl]] += 1

        return M, labels2

    # -------------------------
    # UR/OR/DR/IR/FR/MR (frame or sec bins)
    # -------------------------
    def _segments_to_bins(segments, L, unit="frame", seconds_per_unit=1.0):
        """
        segments: list[[s,e]] in original unit
        L: length of timeline in bins
        unit:
          - frame: bins are frames (1 bin = 1 frame/index)
          - sec  : bins are seconds (1 bin = 1 second)
        """
        bins = np.zeros((L,), dtype=np.uint8)
        if L <= 0:
            return bins

        for s, e in segments:
            s = float(s)
            e = float(e)
            if e <= s:
                continue

            if unit == "frame":
                a = int(np.floor(s))
                b = int(np.ceil(e))
            else:
                # sec
                s_sec = s * seconds_per_unit
                e_sec = e * seconds_per_unit
                a = int(np.floor(s_sec))
                b = int(np.ceil(e_sec))

            if b <= a:
                continue
            a = max(0, min(a, L))
            b = max(0, min(b, L))
            if b > a:
                bins[a:b] = 1

        return bins

    def _events_from_bins(bins):
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

    def _uodifm_for_class(gt_bins, pred_bins):
        """
        返回“长度”（单位=bin数）以及 P/N（也是bin数）
        """
        L = len(gt_bins)
        P = int(gt_bins.sum())
        N = int(L - P)

        gt_events = _events_from_bins(gt_bins)
        pred_events = _events_from_bins(pred_bins)

        UR = DR = FR = 0
        # --- UR/DR/FR：按GT事件 ---
        for (gs, ge) in gt_events:
            sl = pred_bins[gs:ge]
            if sl.sum() == 0:
                DR += (ge - gs)
                continue

            idx = np.where(sl == 1)[0]
            first = gs + int(idx[0])
            last = gs + int(idx[-1])

            # Underfill: start欠填 + end欠填
            UR += (first - gs) + (ge - (last + 1))

            # Fragmentation: GT区间内“中间断裂”的0
            if last > first:
                FR += int((pred_bins[first:last + 1] == 0).sum())

        OR = IR = MR = 0
        # --- OR/IR/MR：按Pred事件 ---
        for (ps, pe) in pred_events:
            # 如果pred event 完全落在GT负类 -> insertion
            if gt_bins[ps:pe].sum() == 0:
                IR += (pe - ps)
                continue

            # pred event 覆盖到的gt event数量
            overlapped = []
            for (gs, ge) in gt_events:
                if max(ps, gs) < min(pe, ge):
                    overlapped.append((gs, ge))

            neg_cnt = int((gt_bins[ps:pe] == 0).sum())  # pred为1但gt为0

            if len(overlapped) <= 1:
                OR += neg_cnt
            else:
                # merge区域：跨多个gt event之间的桥接负类
                first_gt = min(x[0] for x in overlapped)
                last_gt = max(x[1] for x in overlapped)
                inside_s = max(ps, first_gt)
                inside_e = min(pe, last_gt)
                mr = int((gt_bins[inside_s:inside_e] == 0).sum())
                MR += mr
                OR += (neg_cnt - mr)

        return {"UR": UR, "OR": OR, "DR": DR, "IR": IR, "FR": FR, "MR": MR, "P": P, "N": N}

    def _compute_uodifm(gt_by_vid, pred_by_vid, class_names, unit="frame", seconds_per_unit=1.0):
        vids = sorted(set(gt_by_vid.keys()) | set(pred_by_vid.keys()))
        per_cls = {}

        for cls in class_names:
            acc = {"UR": 0, "OR": 0, "DR": 0, "IR": 0, "FR": 0, "MR": 0, "P": 0, "N": 0}

            for vid in vids:
                gts = [g["segment"] for g in gt_by_vid.get(vid, []) if g["label"] == cls]
                prs = [p["segment"] for p in pred_by_vid.get(vid, []) if p["label"] == cls]

                # timeline长度：取gt/pred最大end
                max_end = 0.0
                for s, e in gts:
                    max_end = max(max_end, float(e))
                for s, e in prs:
                    max_end = max(max_end, float(e))

                if max_end <= 0:
                    continue

                if unit == "frame":
                    L = int(np.ceil(max_end))
                else:
                    L = int(np.ceil(max_end * seconds_per_unit))

                if L <= 0:
                    continue

                gt_bins = _segments_to_bins(gts, L, unit=unit, seconds_per_unit=seconds_per_unit)
                pred_bins = _segments_to_bins(prs, L, unit=unit, seconds_per_unit=seconds_per_unit)

                d = _uodifm_for_class(gt_bins, pred_bins)
                for k in acc:
                    acc[k] += d[k]

            P = acc["P"]
            N = acc["N"]
            ur = acc["UR"] / P if P > 0 else 0.0
            dr = acc["DR"] / P if P > 0 else 0.0
            fr = acc["FR"] / P if P > 0 else 0.0
            orr = acc["OR"] / N if N > 0 else 0.0
            ir = acc["IR"] / N if N > 0 else 0.0
            mr = acc["MR"] / N if N > 0 else 0.0

            per_cls[cls] = {
                "UR": ur, "OR": orr, "DR": dr, "IR": ir, "FR": fr, "MR": mr,
                "UR_len": acc["UR"], "OR_len": acc["OR"], "DR_len": acc["DR"], "IR_len": acc["IR"],
                "FR_len": acc["FR"], "MR_len": acc["MR"], "P_len": P, "N_len": N,
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

    # -------------------------
    # run
    # -------------------------
    gt_by_vid, pred_by_vid, total_gt, total_pred, max_end_gt, max_end_pr = _load_gt_pred(
        gt_path, pred_path, subset_name=subset_name
    )

    # overall PRF
    overall = {thr: _overall_prf(gt_by_vid, pred_by_vid, total_gt, thr) for thr in tious}
    overall_avg = {
        "P": float(np.mean([overall[t]["P"] for t in tious])),
        "R": float(np.mean([overall[t]["R"] for t in tious])),
        "F1": float(np.mean([overall[t]["F1"] for t in tious])),
    }

    # per-action PRF csv
    prf_csv = os.path.join(result_dir, f"per_action_prf_{test_mode}.csv")
    with open(prf_csv, "w", encoding="utf-8") as f:
        f.write("action,tiou,TP,FP,FN,P,R,F1\n")
        for cls in class_names:
            for thr in tious:
                d = _per_action_prf(gt_by_vid, pred_by_vid, cls, thr)
                f.write(f"{cls},{thr:.2f},{d['TP']},{d['FP']},{d['FN']},{d['P']:.6f},{d['R']:.6f},{d['F1']:.6f}\n")

    # confusion matrix plot (greens + row normalized)
    cm_thr = float(confusion_tiou)
    cm, cm_labels = _confusion_matrix(gt_by_vid, pred_by_vid, class_names, cm_thr, include_bg=include_bg)
    cm_png = os.path.join(result_dir, f"confusion_matrix_{test_mode}_tiou{cm_thr:.2f}.png")

    cm_float = cm.astype(np.float32)
    row_sum = cm_float.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_float, row_sum, out=np.zeros_like(cm_float), where=(row_sum > 0))

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
    plt.title(f"Confusion Matrix (tIoU={cm_thr:.2f}) - Row Normalized")
    plt.xlabel("Predicted Label")
    plt.ylabel("GT Label")
    plt.xticks(range(len(cm_labels)), cm_labels, rotation=90, fontsize=8)
    plt.yticks(range(len(cm_labels)), cm_labels, fontsize=8)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = float(cm_norm[i, j])
            if v >= float(cm_value_thresh):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)

    plt.tight_layout()
    plt.savefig(cm_png, dpi=200)
    plt.close()

    # UODIFM (UR/OR/DR/IR/FR/MR)
    if uodifm_unit == "sec":
        if seconds_per_unit is None:
            raise ValueError("uodifm_unit='sec' requires seconds_per_unit (e.g. 1/fps or 1.0 if already seconds)")
        sp = float(seconds_per_unit)
    else:
        sp = 1.0  # unused

    per_cls_uodifm, macro_uodifm = _compute_uodifm(
        gt_by_vid, pred_by_vid, class_names, unit=uodifm_unit, seconds_per_unit=sp
    )

    uodifm_csv = os.path.join(result_dir, f"per_action_uodifm_{test_mode}.csv")
    with open(uodifm_csv, "w", encoding="utf-8") as f:
        f.write("action,UR,OR,DR,IR,FR,MR,UR_len,OR_len,DR_len,IR_len,FR_len,MR_len,P_len,N_len,unit\n")
        for cls in class_names:
            d = per_cls_uodifm.get(cls, None)
            if d is None:
                continue
            f.write(
                f"{cls},{d['UR']:.6f},{d['OR']:.6f},{d['DR']:.6f},{d['IR']:.6f},{d['FR']:.6f},{d['MR']:.6f},"
                f"{d['UR_len']},{d['OR_len']},{d['DR_len']},{d['IR_len']},{d['FR_len']},{d['MR_len']},{d['P_len']},{d['N_len']},{uodifm_unit}\n"
            )

    # report lines
    report_lines = []
    report_lines.append("-" * 60)
    report_lines.append("Extra Metrics: Overall Precision/Recall/F1 (micro) @ tIoU=0.3~0.7")
    report_lines.append(f"Total GT = {total_gt}, Total Pred = {total_pred}")
    for thr in tious:
        d = overall[thr]
        report_lines.append(
            f"tIoU={thr:.2f} → P={d['P']:.4f}, R={d['R']:.4f}, F1={d['F1']:.4f} (TP={d['TP']}, FP={d['FP']}, FN={d['FN']})"
        )
    report_lines.append(
        f"Avg(PRF over tIoU=0.3~0.7) → P={overall_avg['P']:.4f}, R={overall_avg['R']:.4f}, F1={overall_avg['F1']:.4f}"
    )

    report_lines.append("-" * 60)
    report_lines.append("Per-action PRF saved (per tIoU):")
    report_lines.append(prf_csv)

    report_lines.append("-" * 60)
    report_lines.append(f"Confusion matrix saved: {cm_png} (Greens + row-normalized, include_bg={include_bg})")

    report_lines.append("-" * 60)
    report_lines.append(f"Temporal Metrics (UR/OR/DR/IR/FR/MR) unit = {uodifm_unit.upper()}")
    if uodifm_unit == "sec":
        report_lines.append(f"seconds_per_unit = {sp:.6f} (unit->sec)")
    report_lines.append(
        f"MACRO(avg over classes): UR={macro_uodifm['UR']:.4f}, OR={macro_uodifm['OR']:.4f}, "
        f"DR={macro_uodifm['DR']:.4f}, IR={macro_uodifm['IR']:.4f}, FR={macro_uodifm['FR']:.4f}, MR={macro_uodifm['MR']:.4f}"
    )
    report_lines.append("Per-action UODIFM saved:")
    report_lines.append(uodifm_csv)

    artifacts = {
        "per_action_prf_csv": prf_csv,
        "per_action_uodifm_csv": uodifm_csv,
        "confusion_png": cm_png,
    }
    return report_lines, artifacts





# import numpy as np
#
# gt_path = "/home/lipei/project/WSDDN/test_results/WEAR/oicrBUAA_0112/fold17/gt_for_anet.json"
# pred_path = "/home/lipei/project/WSDDN/test_results/WEAR/oicrBUAA_0112/fold17/inference_stats_test_full.json"
# result_dir = "/home/lipei/project/WSDDN/test_results/WEAR/oicrBUAA_0112/fold17"
#
# # class_names 必须包含所有动作名（和json中的label一致）
# class_names = [
#     "jogging",
#         "jogging (rotating arms)",
#         "jogging (skipping)",
#         "jogging (sidesteps)",
#         "jogging (butt-kicks)",
#         "stretching (triceps)",
#         "stretching (lunging)",
#         "stretching (shoulders)",
#         "stretching (hamstrings)",
#         "stretching (lumbar rotation)",
#         "push-ups",
#         "push-ups (complex)",
#         "sit-ups",
#         "sit-ups (complex)",
#         "burpees",
#         "lunges",
#         "lunges (complex)",
#         "bench-dips"
# ]
#
# tious = np.linspace(0.3, 0.7, 5)
#
# report_lines, artifacts = evaluate_extra_metrics_and_plots(
#     gt_path=gt_path,
#     pred_path=pred_path,
#     class_names=class_names,
#     tious=tious,
#     result_dir=result_dir,
#     test_mode="test_full",   # 只是文件名后缀，不影响计算
#     subset_name="test",
#     confusion_tiou=0.5,
#     include_bg=True,
#     uodifm_unit="frame",        # 如果你的segment是帧/index
#     # uodifm_unit="sec", seconds_per_unit=1.0  # 如果你的segment是秒
#     cm_value_thresh=0.001,
# )
#
# print("\n".join(report_lines))
# print(artifacts)
