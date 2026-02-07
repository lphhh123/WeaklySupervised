import os
import sys
import torch
import json
import numpy as np
from tqdm import tqdm
import copy

                  
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from core.model import CoLA
from core import utils
from run_cola_hangtime_ddp import map_config_to_cola_cfg, get_inference_data
from WSDDN.HANGTIME.dataset_hangtime_ws import WeaklyHangtimeDataset
from eval_folds_metrics import evaluate_experiment_dir

                                          
                       
CKPT_PATH = "/home/lipei/project/CoLA/output_hangtime_cola_ddp/seed_2022/fold0/model_final.pth"
                     
TMP_EVAL_DIR = "/home/lipei/project/CoLA/output_hangtime_cola_ddp/seed_2024/fold0_debug"
# /home/lipei/project/CoLA/output_hangtime_cola_ddp/seed_2022/fold0/model_final.pth
BASE_CONFIG = {
    "dataset_dir": "/home/lipei/TAL_data/hangtime/",
    "pretrained_dir": "/home/lipei/project/WSDDN/OtherData/HANGTIME/pre_train/CNN1D",
    "result_root": TMP_EVAL_DIR,
    "fps": 50,
    "in_channels": 3,
    "num_classes": 5,
    "clip_sec": 30.0,
    "clip_overlap": 0.5,
    "pretrained_model_name": "CNN1D",
    "training": {"train_backbone": False},
    "cola": {
        "lambda": 0.01, "r_easy": 100, "r_hard": 20, "m": 3, "M": 10,
        "class_thresh": 0.1,          
        "nms_thresh": 0.3
    }
}


# ===========================================

@torch.no_grad()
def run_optimized_inference():
    device = torch.device("cuda:0")
    fold = 0
    os.makedirs(os.path.join(TMP_EVAL_DIR, "fold0"), exist_ok=True)

             
    cfg = map_config_to_cola_cfg(BASE_CONFIG, fold)
    net = CoLA(cfg).to(device)
    net.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    net.eval()

              
    test_dataset = WeaklyHangtimeDataset(
        dataset_dir=BASE_CONFIG["dataset_dir"],
        loso_json=f"loso_sbj_{fold}.json",
        mode="test_window", fps=50, num_sensors=3, normalize=True
    )

    id2name = {v: k for k, v in cfg.CLASS_DICT.items()}

                                         
    mode = "test_full"
    win_size = cfg.NUM_SEGMENTS  # 1500
    final_res = {'version': 'VERSION 1.3', 'results': {}}

    for sbj, window_iter, t_origin in get_inference_data(test_dataset, mode, win_size, win_size // 2):
        video_props = []
        for chunk, _ in window_iter:
            chunk = chunk.to(device)
            video_scores, _, actionness, cas = net(chunk)

                              
            cas_prob = torch.sigmoid(cas[0])  # [1500, 5]
            aness_prob = torch.sigmoid(actionness[0])  # [1500]
            v_prob = torch.sigmoid(video_scores[0]).cpu().numpy()

                               
            cas_suppressed = cas_prob * aness_prob.unsqueeze(1)

            cas_np = np.expand_dims(cas_suppressed.cpu().numpy(), axis=2)
            aness_np = np.tile(aness_prob.cpu().numpy().reshape(-1, 1, 1), (1, 5, 1))

                            
            pred_cats = np.where(v_prob >= BASE_CONFIG['cola']['class_thresh'])[0]
            if len(pred_cats) == 0: pred_cats = np.array([np.argmax(v_prob)])

            prop_dict = utils.get_proposal_dict(cas_np, aness_np, pred_cats, v_prob, win_size, cfg)

            for cls_id, props in prop_dict.items():
                for p in props:
                                                      
                    if p[1] <= 0: continue

                    scale = t_origin / win_size
                                               
                    global_start_f = p[2] * scale
                    global_end_f = p[3] * scale

                    video_props.append([p[0], p[1], global_start_f, global_end_f])

                                  
        sbj_list = []
        if video_props:
            cls_groups = {}
            for p in video_props: (cls_groups.setdefault(int(p[0]), [])).append(p)
            for cid, props in cls_groups.items():
                keep = utils.nms(props, BASE_CONFIG['cola']['nms_thresh'])
                for k in keep:
                    sbj_list.append({
                        'label': id2name.get(cid, str(cid)),
                        'score': float(k[1]),
                        'segment': [round(float(k[2]) / 50.0, 2), round(float(k[3]) / 50.0, 2)]              
                    })
                                      
            sbj_list = sorted(sbj_list, key=lambda x: x['score'], reverse=True)[:100]
        final_res['results'][sbj] = sbj_list

                
    pred_path = os.path.join(TMP_EVAL_DIR, "fold0", f"predictions_{mode}.json")
    with open(pred_path, 'w') as f:
        json.dump(final_res, f, indent=2)

                                           
    shutil_copy = os.path.join(TMP_EVAL_DIR, "fold0", "predictions_test_window.json")
    with open(shutil_copy, 'w') as f:
        json.dump(final_res, f, indent=2)

                               
    orig_gt = os.path.join(os.path.dirname(CKPT_PATH), "gt_for_anet.json")
    target_gt = os.path.join(TMP_EVAL_DIR, "fold0", "gt_for_anet.json")
    if os.path.exists(orig_gt):
        import shutil
        shutil.copy(orig_gt, target_gt)

    print(f"\n✅ 推理完成，JSON 已存至: {pred_path}")
    print("-" * 50)
    print("🚀 开始自动调用评估脚本...")

               
    evaluate_experiment_dir(
        TMP_EVAL_DIR,
        tiou_thresholds=(0.3, 0.4, 0.5, 0.6, 0.7),
        use_conf_thresh=True,
        conf_thresh_override=0
    )

                
    summary_path = os.path.join(TMP_EVAL_DIR, "metrics_summary_all_folds.json")
    with open(summary_path, 'r') as f:
        stats = json.load(f)['modes']['test_full']['mean_over_folds']
        print("\n" + "=" * 20 + " 修正后 Fold0 结果 " + "=" * 20)
        print(f"mAP@Avg: {stats['mAP_mean']:.4f}")
        print(f"F1-Macro: {stats['F1_macro']:.4f}")
        print(f"IR (Insertion Ratio): {stats['UODIFM']['IR']:.4f} (期望大幅下降)")
        print("=" * 50)


if __name__ == "__main__":
    run_optimized_inference()