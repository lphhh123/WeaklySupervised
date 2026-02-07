# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()

cfg.TRAIN_BACKBONE = False
cfg.NUM_EPOCHS = 80
cfg.BATCH_SIZE = 32

cfg.LR_BASE = 1e-4
cfg.LR_STEP_SIZE = 10
cfg.LR_GAMMA = 0.9
cfg.WEIGHT_DECAY = 0.0005

cfg.GPU_ID = '0'
cfg.NUM_WORKERS = 4
cfg.SEED = 2022

cfg.MODAL = 'imu'
cfg.USE_AIRPODS = True
cfg.FEATS_DIM = 36
cfg.NUM_CLASSES = 30
cfg.FEATS_FPS = 50
cfg.NUM_SEGMENTS = 2048
cfg.UP_SCALE = 1

# cfg.DATA_PATH = '/data1/WSTAL/all_6_30_3'
cfg.DATA_PATH = '/home/lipei/XRFV2'
cfg.GT_PATH = os.path.join(cfg.DATA_PATH, 'imu_annotations.json')
cfg.TEST_DATA_ROOT = '/home/lipei/WWADL/imu'
cfg.DATASET_NAME = 'XRFV2'
cfg.BACKBONE_TYPE = 'cnn1d'
cfg.PRETRAINED_PATH = '/home/lipei/project/CoLA/output_ddp/classifier_model/classifier_best.pth'
cfg.OUTPUT_PATH = '/home/lipei/project/CoLA/output_ddp'
cfg.MODEL_PATH = '/home/lipei/project/CoLA/output_ddp/classifier_model'

cfg.LAMBDA = 0.01
cfg.R_EASY = 15
cfg.R_HARD = 20
cfg.m = 1
cfg.M = 10

cfg.CLASS_THRESH = 0
cfg.NMS_THRESH = 0.4
cfg.CAS_THRESH = np.arange(0.1, 0.4, 0.05)
cfg.ANESS_THRESH = np.arange(0.1, 0.925, 0.025)
cfg.TIOU_THRESH = np.array([0.3, 0.4, 0.5, 0.6, 0.7])

cfg.TEST_FREQ = 100
cfg.PRINT_FREQ = 20

cfg.UP_SCALE = 1

cfg.FEATS_FPS = 50
cfg.NUM_SEGMENTS = 2048

cfg.CLASS_DICT = {
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
