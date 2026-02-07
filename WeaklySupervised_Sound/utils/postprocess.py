from __future__ import annotations

from typing import Dict, List

import numpy as np


def frame_probs_to_segments(probs: np.ndarray, fps: float, threshold: float = 0.1, min_duration: float = 0.2):
    mask = probs > threshold
    if not np.any(mask):
        return []
    diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    segments = []
    for s, e in zip(starts, ends):
        duration = (e - s) / fps
        if duration < min_duration:
            continue
        score = float(np.mean(probs[s:e]))
        segments.append([s / fps, e / fps, score])
    return segments


def frame_probs_to_segments_v2(
    probs: np.ndarray,
    fps: float,
    threshold: float = 0.1,
    min_duration: float = 0.5,
    sigma: float = 2.0,
):
    if sigma > 0:
        probs = gaussian_smooth(probs, sigma)
    return frame_probs_to_segments(probs, fps, threshold=threshold, min_duration=min_duration)


def gaussian_smooth(signal: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return np.convolve(signal, kernel, mode="same")


def temporal_nms(detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    if not detections:
        return []
    final_results = []
    labels = set([d["label"] for d in detections])
    for label in labels:
        label_dets = [d for d in detections if d["label"] == label]
        label_dets = sorted(label_dets, key=lambda x: x["score"], reverse=True)
        keep = []
        while label_dets:
            curr = label_dets.pop(0)
            keep.append(curr)
            remaining = []
            for det in label_dets:
                s1, e1 = curr["segment"]
                s2, e2 = det["segment"]
                inter = max(0, min(e1, e2) - max(s1, s2))
                union = (e1 - s1) + (e2 - s2) - inter
                iou = inter / union if union > 0 else 0
                if iou < iou_threshold:
                    remaining.append(det)
            label_dets = remaining
        final_results.extend(keep)
    return final_results


def soft_nms_functional(dets: np.ndarray, sigma: float = 0.5, thresh: float = 0.001):
    if len(dets) == 0:
        return []
    dets = dets.copy()
    scores = dets[:, 2]
    keep = []
    while dets.shape[0] > 0:
        max_idx = np.argmax(scores)
        best = dets[max_idx]
        keep.append(best)
        dets = np.delete(dets, max_idx, axis=0)
        scores = np.delete(scores, max_idx, axis=0)
        if dets.shape[0] == 0:
            break
        ious = np.array([_segment_iou(best[:2], det[:2]) for det in dets])
        scores = scores * np.exp(-(ious ** 2) / sigma)
        keep_mask = scores > thresh
        dets = dets[keep_mask]
        scores = scores[keep_mask]
    return keep


def _segment_iou(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def build_anet_predictions(results: Dict[str, List[Dict]], version: str = "VERSION 1.3") -> Dict:
    return {
        "version": version,
        "results": results,
        "external_data": {},
    }
