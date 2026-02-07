import torch
import json
import os
import copy
from tqdm import tqdm
import numpy as np
from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTest


class LOSOInferenceEngine:
    def __init__(self, config, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fps = 50.0
        self.n_classes = config["training"]["num_classes"]
        self.config = config

    def load_model(self, model_path):
        from models.DCASE_CRNN_XRFV2 import CRNN
        model = CRNN(n_in_channel=1, nclass=self.n_classes, **self.config["model"]["params"]).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def temporal_nms(self, detections, iou_threshold=0.3):
        """
        Apply temporal NMS to detected segments and merge overlaps.
        """
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

    def post_process_to_dict(self, probs, threshold, offset_frames, id_to_action):
        n_class, n_frames = probs.shape
        video_annotations = []

        for cls_idx in range(n_class):
            mask = probs[cls_idx, :] > threshold
            if not np.any(mask): continue

            diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            label_name = id_to_action.get(str(cls_idx), str(cls_idx))

            for s, e in zip(starts, ends):
                abs_s = float(s + offset_frames)
                abs_e = float(e + offset_frames)
                score = float(probs[cls_idx, s:e].mean())

                video_annotations.append({
                    "segment": [round(abs_s / self.fps, 2), round(abs_e / self.fps, 2)],
                    "label": label_name,
                    "score": round(score, 4)
                })
        return video_annotations

    def run_inference(self, model, test_ds, mode="window"):
        results = {}
        id_to_action = test_ds.id_to_action

        with torch.no_grad():
            dataset_gen = test_ds.dataset()

            for file_name, data_iter in tqdm(dataset_gen, desc=f"  [{mode}]", leave=False):
                video_annotations = []

                for clip_dict, info in data_iter:
                    data = clip_dict['imu'].unsqueeze(0).to(self.device)

                    # print(f"DEBUG: {file_name} input shape: {data.shape}")

                    strong_out, _ = model(data)
                    pred_np = strong_out.squeeze(0).cpu().numpy()

                    offset_frames = 0
                    if isinstance(info, (list, tuple, np.ndarray)):
                        offset_frames = int(info[0])
                    elif isinstance(info, dict):
                        offset_frames = int(info.get('offset', 0))

                    clip_results = self.post_process_to_dict(
                        pred_np,
                        threshold=0.3,
                        offset_frames=offset_frames,
                        id_to_action=id_to_action
                    )
                    video_annotations.extend(clip_results)

                if mode == "window":
                    video_annotations = self.temporal_nms(video_annotations)

                video_annotations.sort(key=lambda x: x["segment"][0])
                results[file_name] = video_annotations

        return results


def run_loso_test():
    base_config = {
        "path": {
            "test_dataset_path": "",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "result_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_loso_2022"
        },
        "model": {
            "params": {
                "rnn_type": "BGRU", "n_RNN_cell": 128, "n_layers_RNN": 2,
                "dropout": 0.5, "attention": True, "specaugm_t_l": 10, "specaugm_f_l": 2
            }
        },
        "training": {"num_classes": 30, "use_airpods": True},
        "testing": {"window_size": 10, "hop_size": 5}
    }

    ckpt_root = "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/xrfv2_dcase_loso_2022"
    subjects = ["all_6_30_3", "all_5_30_3", "all_4_30_3", "all_2_30_3"]

    engine = LOSOInferenceEngine(base_config)

    for sub in subjects:
        print(f"\n🚀 Processing Subject: {sub}")
        sub_dir = os.path.join(ckpt_root, sub)
        model_files = [f for f in os.listdir(sub_dir) if f.endswith('.pth')]
        if not model_files:
            print(f"⚠️ No model found in {sub_dir}")
            continue

        model_path = os.path.join(sub_dir, model_files[0])
        current_model = engine.load_model(model_path)
        save_dir = os.path.join(base_config["path"]["result_path"], sub)
        os.makedirs(save_dir, exist_ok=True)

        for mode_name, win_size in [("window", 10), ("full", 1000)]:
            current_cfg = copy.deepcopy(base_config)
            current_cfg["path"]["test_dataset_path"] = os.path.join("/home/lipei/", sub)
            current_cfg["testing"]["window_size"] = win_size

            if mode_name == "full":
                current_cfg["testing"]["hop_size"] = win_size

            ds = WeaklySupervisedXRFV2DatasetTest(config=current_cfg, use_airpods=True)
            res = engine.run_inference(current_model, ds, mode=mode_name)

            output_file = os.path.join(save_dir, f"predictions_test_{mode_name}.json")
            with open(output_file, "w") as f:
                json.dump({
                    "version": "VERSION 1.3",
                    "results": res,
                    "external_data": {}
                }, f, indent=4)

        print(f" {sub} results saved.")


if __name__ == "__main__":
    run_loso_test()
