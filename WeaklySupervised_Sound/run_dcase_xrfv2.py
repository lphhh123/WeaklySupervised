import os
import json
import torch
import numpy as np
from tqdm import tqdm
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTest
from models.DCASE_CRNN_XRFV2 import CRNN


def temporal_nms(detections, iou_threshold=0.3):
    if not detections: return []
    final_results = []
    labels = set([d['label'] for d in detections])
    for label in labels:
        label_dets = [d for d in detections if d['label'] == label]
        label_dets = sorted(label_dets, key=lambda x: x['score'], reverse=True)
        keep = []
        while label_dets:
            curr = label_dets.pop(0)
            keep.append(curr)
            remaining = []
            for det in label_dets:
                s1, e1 = curr['segment']
                s2, e2 = det['segment']
                inter = max(0, min(e1, e2) - max(s1, s2))
                union = (e1 - s1) + (e2 - s2) - inter
                iou = inter / union if union > 0 else 0
                if iou < iou_threshold:
                    remaining.append(det)
            label_dets = remaining
        final_results.extend(keep)
    return final_results


def post_process_to_dict(predictions, threshold, offset_frames, id_to_action):
    n_class, n_frames = predictions.shape
    video_annotations = []

    for cls_idx in range(n_class):
        probs = predictions[cls_idx, :]
        mask = probs > threshold

        if not np.any(mask): continue

        diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        label_name = id_to_action.get(str(cls_idx), str(cls_idx))
        for s, e in zip(starts, ends):
            abs_s = int(s + offset_frames)
            abs_e = int(e + offset_frames)
            score = float(probs[s:e].mean())

            video_annotations.append({
                "segment": [abs_s, abs_e],
                "label": label_name,
                "score": round(score, 4)
            })
    return video_annotations


def run_pure_test(config, checkpoint_path, test_mode="window"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRNN(n_in_channel=1, nclass=config["training"]["num_classes"], **config["model"]["params"]).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Error: model checkpoint not found {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_config = copy.deepcopy(config)
    if test_mode == "full":
        test_config["testing"]["window_size"] = 2000
        test_config["testing"]["hop_size"] = 2000

    test_ds = WeaklySupervisedXRFV2DatasetTest(config=test_config, use_airpods=config["training"]["use_airpods"])
    id_to_action = test_ds.id_to_action

    results = {}
    print(f"\nRunning inference ({test_mode} mode)...")

    with torch.no_grad():
        for file_name, data_iter in tqdm(test_ds.dataset(), desc=f"Mode: {test_mode}"):
            video_annotations = []
            for clip_dict, info in data_iter:
                data = clip_dict['imu'].unsqueeze(0).to(device)

                # print(f"Input Mean: {data.mean().item():.4f}, Std: {data.std().item():.4f}")

                strong_out, _ = model(data)

                # print(f"Logits Max: {strong_out.max().item():.4f}, Min: {strong_out.min().item():.4f}")

                T = 0.5
                pred_probs = torch.sigmoid(strong_out / T).squeeze(0).cpu().numpy()

                offset_frames = 0
                if isinstance(info, (list, tuple, np.ndarray)):
                    offset_frames = int(info[0])
                elif isinstance(info, dict):
                    offset_frames = int(info.get('offset', 0))

                clip_results = post_process_to_dict(
                    pred_probs,
                    threshold=0.35,
                    offset_frames=offset_frames,
                    id_to_action=id_to_action
                )
                video_annotations.extend(clip_results)

            if test_mode == "window":
                video_annotations = temporal_nms(video_annotations, iou_threshold=0.3)

            video_annotations.sort(key=lambda x: x["segment"][0])
            results[file_name] = video_annotations

    save_path = os.path.join(config["path"]["result_path"], f"predictions_test_{test_mode}.json")
    with open(save_path, 'w') as f:
        json.dump({"version": "VERSION 1.3", "results": results, "external_data": {}}, f, indent=4)
    print(f"Results saved to: {save_path}")


def main():
    base_config = {
        "path": {
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/xrfv2_dcase_2024",
            "result_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_2022"
        },
        "model": {
            "params": {
                "rnn_type": "BGRU", "n_RNN_cell": 128, "n_layers_RNN": 2,
                "dropout": 0.5, "attention": True, "specaugm_t_l": 10, "specaugm_f_l": 2
            }
        },
        "training": {"use_airpods": True, "num_classes": 30},
        "testing": {"window_size": 20, "hop_size": 10}
    }

    os.makedirs(base_config["path"]["result_path"], exist_ok=True)
    exp_name = "xrfv2_dcase_crnn_v1"
    best_ckpt_path = os.path.join(base_config["path"]["checkpoint_path"], f"{exp_name}_best.pth")

    run_pure_test(base_config, best_ckpt_path, test_mode="window")
    # run_pure_test(base_config, best_ckpt_path, test_mode="full")


if __name__ == "__main__":
    main()
