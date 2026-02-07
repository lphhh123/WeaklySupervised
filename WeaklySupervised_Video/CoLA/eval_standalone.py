import os
import sys
import torch
import numpy as np
import json
from terminaltables import AsciiTable

        
import core.utils as utils
from core.model import CoLA
from core.config_xrfv2 import cfg
from core.dataset_xrfv2 import XRFV2Dataset
from main_cola import test_all


def run_inference():
    print("======== Re-inference ========")

                                 
                                              
    CORRECT_DICT = {
        "Stretching": 0,
        "Pouring Water": 1,
        "Writing": 2,
        "Cutting Fruit": 3,
        "Eating Fruit": 4,
        "Taking Medicine": 5,
        "Drinking Water": 6,
        "Sitting Down": 7,
        "Turning On/Off Eye Protection Lamp": 8,
        "Opening/Closing Curtains": 9,
        "Opening/Closing Windows": 10,
        "Typing": 11,
        "Opening Envelope": 12,
        "Throwing Garbage": 13,
        "Picking Fruit": 14,
        "Picking Up Items": 15,
        "Answering Phone": 16,
        "Using Mouse": 17,
        "Wiping Table": 18,
        "Writing on Blackboard": 19,
        "Washing Hands": 20,
        "Using Phone": 21,
        "Reading": 22,
        "Watering Plants": 23,
        "Walking": 24,
        "Getting Out of Bed": 25,
        "Standing Up": 26,
        "Lying Down": 27,
        "Standing Still": 28,
        "Lying Still": 29
    }
    cfg.CLASS_DICT = CORRECT_DICT
    cfg.NUM_CLASSES = len(CORRECT_DICT)

             
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

                 
                                                   
    ckpt_path = os.path.join('output_ddp', 'cnn1d_0124_1710', 'checkpoints', 'model_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"Model checkpoint not found: {ckpt_path}")
                   
        # ckpt_path = ...
        return

    print(f"Loading Model from: {ckpt_path}")

             
    net = CoLA(cfg)
    net = net.cuda()
                                                 
    state_dict = torch.load(ckpt_path)

                    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    print("Model loaded successfully")

              
    test_loader = torch.utils.data.DataLoader(
        XRFV2Dataset(
            mode='test',
            modal=cfg.MODAL,
            num_segments=cfg.NUM_SEGMENTS,
            class_dict=cfg.CLASS_DICT,
            seed=cfg.SEED,
            supervision='weak'
        ),
        batch_size=1,
        shuffle=False, num_workers=4
    )

                     
    print("Running Inference...")
    test_info = {"step": [], "test_acc": [], "average_mAP": []}
    for i in cfg.TIOU_THRESH:
        test_info[f"mAP@{i:.1f}"] = []

                                                       
                                                 
    cfg.OUTUT_PATH = 'output_ddp'

    mAP_50, mAP_AVG = test_all(net, cfg, test_loader, test_info, 0, None)

    print(f"\n======== Re-evaluation results ========")
    print(utils.table_format(test_info, cfg.TIOU_THRESH, '[CoLA] XRFV2 Re-Inference'))
    print(f"Results saved to: {os.path.join(cfg.OUTPUT_PATH, 'result.json')}")


if __name__ == '__main__':
    run_inference()
