# run_dcase_xrfv2_loso.py
import os
import torch
import json
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset_xrfv2 import (
    WeaklySupervisedXRFV2DatasetTrain,
    WeaklySupervisedXRFV2DatasetTest,
    FullBackboneWrapper1D
)
from models.DCASE_CRNN import CRNN
from tool import ANETdetection
from OtherData.utils import set_seed


                                         
class DCASEWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        f_p, _ = self.model(x)
        return f_p


def frame_probs_to_segments(probs, fps, threshold=0.5):
    T, C = probs.shape
    segments = [[] for _ in range(C)]
    for c in range(C):
        binary = probs[:, c] > threshold
        diff = np.diff(np.concatenate(([0], binary.astype(int), [0])))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) / fps >= 0.1:
                segments[c].append([s / fps, e / fps, np.mean(probs[s:e, c])])
    return segments


def train_fold(config, fold_idx, device):
    model = CRNN(n_in_channel=36, nclass=30, **config["model_args"]).to(device)
                      
    train_ds = WeaklySupervisedXRFV2DatasetTrain(
        config["subject_path"],
        config["mapping"],
        split="train",
        use_airpods=True                       
    )
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.BCELoss()

    print(f"--- Training Fold {fold_idx} ---")
    for epoch in range(50):
        model.train()
        for x, _, y in loader:
                                             
            if x.shape[1] != 36:
                x = x.transpose(1, 2)
                               
            _, p = model(x.to(device))
            loss = crit(p, y.to(device))
            opt.zero_grad();
            loss.backward();
            opt.step()

    ckpt = f"fold_{fold_idx}.pth"
    torch.save(model.state_dict(), ckpt)
    return ckpt


@torch.no_grad()
def test_fold(config, ckpt, fold_idx, mode, device):
    model = CRNN(n_in_channel=36, nclass=30, **config["model_args"]).to(device)
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    test_ds = WeaklySupervisedXRFV2DatasetTest(config, modality='imu', use_airpods=True)
    # ...
    eval_model = FullBackboneWrapper1D(DCASEWrapper(model), 150, 75, 30).to(device) if mode == "full" else model

    results = {}
    for fname, data_iter in tqdm(test_ds.dataset(), desc=f"Fold{fold_idx}-{mode}"):
        v_res = []
        for clip_dict, seg in data_iter:
            x = clip_dict['imu'].unsqueeze(0).to(device)  # [1, 30, T]
            if mode == "full":
                p = eval_model(x)
            else:
                p, _ = eval_model(x)

            p = torch.nn.functional.interpolate(p, size=x.shape[-1], mode='linear').squeeze(0).T.cpu().numpy()
            segs = frame_probs_to_segments(p, 50)
            offset = seg[0] / 50.0
            for c_idx, s_list in enumerate(segs):
                name = test_ds.id_to_action.get(str(c_idx), f"act_{c_idx}")
                for s, e, sc in s_list:
                    v_res.append({"label": name, "score": float(sc), "segment": [s + offset, e + offset]})
        results[fname] = v_res

    out_json = f"pred_{mode}_{fold_idx}.json"
    with open(out_json, 'w') as f:
        json.dump({"results": results}, f)

    evaluator = ANETdetection(test_ds.eval_gt, out_json, tiou_thresholds=np.linspace(0.3, 0.7, 5), verbose=False)
    return evaluator.evaluate()[1]


if __name__ == "__main__":
    paths = ["/home/lipei/all_6_30_3/", "/home/lipei/all_5_30_3/", "/home/lipei/all_4_30_3/", "/home/lipei/all_2_30_3/"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = {
        "mapping": "/home/lipei/project/WSDDN/label_mapping.json",
        "path": {"test_dataset_path": "", "dataset_root_path": "/home/lipei/WWADL/"},        
        "model_args": {"cnn_integration": False,
                       "cnn_kwargs": {"pooling": [[2, 1], [2, 1], [2, 1], [1, 1], [1, 1], [1, 1], [1, 1]]}}
    }

    for i, p in enumerate(paths):
        conf["subject_path"] = p
        conf["path"]["test_dataset_path"] = p
        ckpt = train_fold(conf, i, device)
        mAP_win = test_fold(conf, ckpt, i, "window", device)
        mAP_full = test_fold(conf, ckpt, i, "full", device)
        print(f"Fold {i} Results: Window={mAP_win:.4f}, Full={mAP_full:.4f}")