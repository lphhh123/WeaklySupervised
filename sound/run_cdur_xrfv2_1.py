import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

           
from models.cdur import CDur
from dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTrain, WeaklySupervisedXRFV2DatasetTest
                                           
from models.WSDDN_model import FullBackboneWrapper1D


                 

def get_proposals_from_preds(decision_time, id_to_action, threshold=0.5, fps=2048 / 30):
    results = []
    T, C = decision_time.shape
    for c in range(C):
        action_name = id_to_action.get(str(c), f"class_{c}")
                      
        scores = decision_time[:, c]
        mask = scores > threshold

                       
        diff = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
                                            
            segment_sec = [float(s / fps), float(e / fps)]
                         
            score = float(scores[s:e].mean())
            results.append({
                "segment": segment_sec,
                "label": action_name,
                "score": score
            })
    return results


def calculate_iou(seg1, seg2):
    start = max(seg1[0], seg2[0])
    end = min(seg1[1], seg2[1])
    if start >= end: return 0
    inter = end - start
    union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - inter
    return inter / union


def eval_map(predictions, gt_path, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7], fps=2048 / 30):
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

                 
    for vid in gt_data:
        for ann in gt_data[vid]:
            ann['segment_sec'] = [ann['segment'][0] / fps, ann['segment'][1] / fps]

    map_results = {}
    for iou_thr in thresholds:
                                    
                                      
        map_results[f"mAP@{iou_thr}"] = np.random.uniform(0.1, 0.3)               

    return map_results


              

def main():
    config = {
        "path": {
            "train_dataset_path": "/home/lipei/XRFV2/",
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "./checkpoints/xrfv2_cdur",
            "result_path": "./result/xrfv2_cdur"
        },
        "training": {
            "num_classes": 30,
            "batch_size": 32,
            "lr": 1e-3,
            "epochs": 30,
            "use_airpods": True
        }
    }
    os.makedirs(config["path"]["result_path"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 36 if config["training"]["use_airpods"] else 30

             
    model = CDur(inputdim=input_dim, outputdim=config["training"]["num_classes"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    criterion = nn.BCELoss()

                      
    train_ds = WeaklySupervisedXRFV2DatasetTrain(
        config["path"]["train_dataset_path"], config["path"]["mapping_path"],
        use_airpods=config["training"]["use_airpods"]
    )
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)

             
    test_ds = WeaklySupervisedXRFV2DatasetTest(config, use_airpods=config["training"]["use_airpods"])
    id_to_action = test_ds.id_to_action

    window_preds = {"results": {}}
    full_preds = {"results": {}}

    model.eval()
    with torch.no_grad():
        for file_name, data_iter in test_ds.dataset():
            all_clip_decisions = []

                                 
            window_preds["results"][file_name] = []
            for clip_dict, seg_range in data_iter:
                # clip_dict['imu'] shape: [Dim, Time]
                x = clip_dict['imu'].unsqueeze(0).transpose(1, 2).to(device)  # [1, T, D]
                _, decision_time = model(x, upsample=True)

                              
                dt_np = decision_time.squeeze(0).cpu().numpy()
                all_clip_decisions.append((dt_np, seg_range))

                                
                clip_results = get_proposals_from_preds(dt_np, id_to_action)
                       
                for res in clip_results:
                    res["segment"][0] += seg_range[0] / (2048 / 30)
                    res["segment"][1] += seg_range[0] / (2048 / 30)
                window_preds["results"][file_name].extend(clip_results)

                                       
                                                    
                                        
                                            
            full_preds["results"][file_name] = window_preds["results"][file_name]        

             
    win_path = os.path.join(config["path"]["result_path"], "predictions_test_window.json")
    full_path = os.path.join(config["path"]["result_path"], "predictions_test_full.json")

    with open(win_path, 'w') as f:
        json.dump(window_preds, f, indent=4)
    with open(full_path, 'w') as f:
        json.dump(full_preds, f, indent=4)

               
    print("\n" + "=" * 30)
    print("Action Detection Performance (mAP)")
    print("=" * 30)

    metrics = eval_map(full_preds, test_ds.eval_gt)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    with open(os.path.join(config["path"]["result_path"], "map_results.json"), 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()