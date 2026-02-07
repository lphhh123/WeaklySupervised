import os
import json
import torch
import numpy as np
import glob
from tqdm import tqdm

                
from tool import load_label_mapping
from models.DCASE_CRNN_XRFV2 import CRNN
from dataset.dataset_XRFV2_loso import WeaklySupervisedXRFV2DatasetTest


                                                               

def temporal_nms(predictions, thresh=0.4):
    if len(predictions) == 0: return []
    df = sorted(predictions, key=lambda x: x['score'], reverse=True)
    keep = []
    while len(df) > 0:
        m = df.pop(0)
        keep.append(m)
        remaining = []
        for n in df:
            if m['label'] == n['label']:
                s = max(m['segment'][0], n['segment'][0])
                e = min(m['segment'][1], n['segment'][1])
                inter = max(0, e - s)
                union = (m['segment'][1] - m['segment'][0]) + (n['segment'][1] - n['segment'][0]) - inter
                if union > 0 and (inter / union) >= thresh: continue
            remaining.append(n)
        df = remaining
    return keep


def post_process_for_anet(probs, id_to_label, threshold=0.3, offset_frame=0):
    n_class, n_frames = probs.shape
    results = []
    for cls_idx in range(n_class):
        label_name = id_to_label.get(cls_idx, "unknown")
        mask = probs[cls_idx, :] > threshold
        start_f = None
        for i in range(n_frames):
            if mask[i] and start_f is None:
                start_f = i
            elif not mask[i] and start_f is not None:
                results.append({
                    "segment": [float(start_f + offset_frame), float(i + offset_frame)],
                    "label": label_name,
                    "score": float(probs[cls_idx, start_f:i].max())
                })
                start_f = None
        if start_f is not None:
            results.append({
                "segment": [float(start_f + offset_frame), float(n_frames + offset_frame)],
                "label": label_name,
                "score": float(probs[cls_idx, start_f:].max())
            })
    return results


                                                                  

def run_inference_only(config, checkpoint_path, test_mode="window"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, id_to_label = load_label_mapping(config["path"]["mapping_path"])

          
    model = CRNN(n_in_channel=1, nclass=config["training"]["num_classes"], **config["model"]["params"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    config["training"]["test_mode"] = test_mode
    test_ds = WeaklySupervisedXRFV2DatasetTest(config=config, use_airpods=config["training"]["use_airpods"])

          
    prediction_output = {"results": {}, "version": "VERSION 1.3", "external_data": {}}

    print(f"\n[Running Inference] Mode: {test_mode}")

    with torch.no_grad():
        for file_name, data_iter in tqdm(test_ds.dataset(), desc=f"Predicting {test_mode}"):
            video_raw_segments = []
            for clip_dict, _ in data_iter:
                data = clip_dict['imu'].unsqueeze(0).to(device)
                strong_out, _ = model(data)

                res = strong_out.squeeze(0).cpu()
                if res.shape[0] == config["training"]["num_classes"]:
                    probs = torch.sigmoid(res).numpy()
                else:
                    probs = torch.sigmoid(res).numpy().T

                offset = clip_dict.get('offset', 0)
                clip_results = post_process_for_anet(probs, id_to_label, threshold=0.2, offset_frame=offset)
                video_raw_segments.extend(clip_results)

                        
            prediction_output["results"][file_name] = temporal_nms(video_raw_segments, thresh=0.4)

                
    file_name = f"pred_{test_mode}.json"
    save_path = os.path.join(config["path"]["result_path"], file_name)
    with open(save_path, 'w') as f:
        json.dump(prediction_output, f, indent=4)

    print(f"Successfully saved: {save_path}")


                                                              

def main():
            
    checkpoint_dir = "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/xrfv2_dcase_2022"
    result_dir = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_dcase_2022_test"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

            
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not ckpt_files:
        print(f"Error: No .pth found in {checkpoint_dir}")
        return
    best_ckpt = ckpt_files[0]

    config = {
        "path": {
            "test_dataset_path": "/home/lipei/all_6_30_3",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "dataset_root_path": "/home/lipei/WWADL/",
            "result_path": result_dir
        },
        "model": {
            "params": {"rnn_type": "BGRU", "n_RNN_cell": 128, "n_layers_RNN": 2, "dropout": 0.5, "attention": True}},
        "training": {"use_airpods": True, "num_classes": 30}
    }

               
    run_inference_only(config, best_ckpt, test_mode="window")
    run_inference_only(config, best_ckpt, test_mode="full")

    print("\nAll tasks completed. JSON files are ready.")


if __name__ == "__main__":
    main()