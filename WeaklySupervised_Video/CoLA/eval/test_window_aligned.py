import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm

      
sys.path.append(os.getcwd())
from core.model import CoLA
from core.config_xrfv2 import cfg
import core.utils as utils
from core.dataset_xrfv2 import XRFV2Dataset

                                          
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "output_ddp", 'test_window')
                       
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'output_ddp', "cnn1d_0124_1710", 'checkpoints', 'model_best.pth')

WINDOW_SIZE = 2048
STRIDE = 512


# ===========================================

@torch.no_grad()
def main():
                       
    net = CoLA(cfg).cuda()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace('module.', '')
        if new_key.startswith('actionness_backbone'):
            new_key = new_key.replace('actionness_backbone', 'actionness_module.backbone')
        elif new_key.startswith('actionness_adapter'):
            new_key = new_key.replace('actionness_adapter', 'actionness_module.adapter')
        elif new_key.startswith('actionness_f_cls'):
            new_key = new_key.replace('actionness_f_cls', 'actionness_module.f_cls')
        state_dict[new_key] = v

    net.load_state_dict(state_dict, strict=False)
    net.eval()
    print(f"Loaded model from {CHECKPOINT_PATH}")

               
    dataset = XRFV2Dataset(mode='test', modal=cfg.MODAL, num_segments=WINDOW_SIZE,
                           class_dict=cfg.CLASS_DICT, supervision='weak')

    final_res = {'method': 'Aligned_Window_Inference', 'results': {}}

                   
    for vid_name, window_iter, t_origin in tqdm(dataset.dataset_windowed(WINDOW_SIZE, STRIDE), desc="Videos"):
        video_all_proposals = []

        for chunk_tensor, win_start_f in window_iter:
            chunk_tensor = chunk_tensor.cuda()
            video_scores, _, actionness, cas = net(chunk_tensor)

                           
            score_np = video_scores[0].cpu().numpy()
            max_s = np.max(score_np)
            pred_categories = np.where(score_np >= max(max_s * 0.4, cfg.CLASS_THRESH))[0]
            if len(pred_categories) == 0:
                pred_categories = np.array([np.argmax(score_np)])

            num_pred = len(pred_categories)

                                                                        
                                                     
            cas_np = cas[0].cpu().numpy()[:, pred_categories]
            cas_np = np.expand_dims(cas_np, axis=2)

                                                           
            aness_raw = actionness[0].cpu().numpy()
            aness_np = np.tile(aness_raw.reshape(-1, 1, 1), (1, num_pred, 1))
            # ===================================================================

                                     
            prop_dict = utils.get_proposal_dict(cas_np, aness_np, pred_categories, score_np, WINDOW_SIZE, cfg)

                     
            for cls_id, props in prop_dict.items():
                for p in props:
                    # p: [cls, score, local_start, local_end]
                    video_all_proposals.append([p[0], p[1], win_start_f + p[2], win_start_f + p[3]])

                         
        if not video_all_proposals:
            final_res['results'][vid_name] = []
            continue

        cls_groups = {}
        for p in video_all_proposals:
            cid = int(p[0])
            if cid not in cls_groups: cls_groups[cid] = []
            cls_groups[cid].append(p)

        video_final_json = []
        class_idx2name = {v: k for k, v in cfg.CLASS_DICT.items()}
        for cid, props in cls_groups.items():
            keep_props = utils.nms(props, cfg.NMS_THRESH)
            for kp in keep_props:
                video_final_json.append({
                    'label': class_idx2name[cid],
                    'score': float(kp[1]),
                    'segment': [float(kp[2]), float(kp[3])]
                })

        final_res['results'][vid_name] = video_final_json

                                
    save_dir = os.path.join(RESULTS_ROOT)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'predictions_test_window_6_2.json')

    with open(save_path, 'w') as f:
        json.dump(final_res, f)
    print(f"Window inference finished. Saved to: {save_path}")


if __name__ == '__main__':
    main()
