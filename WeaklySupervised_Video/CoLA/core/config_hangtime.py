import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()

              
cfg.GPU_ID = '0'
cfg.SEED = 2024
cfg.NUM_WORKERS = 4

                            
cfg.DATASET_NAME = 'HANGTIME'
cfg.BACKBONE_TYPE = 'cnn1d'
cfg.MODAL = 'imu'
cfg.FEATS_FPS = 50
cfg.FEATS_DIM = 3
cfg.NUM_CLASSES = 5
cfg.RAW_WINDOW_SIZE = 1500
cfg.DOWNSAMPLE_RATE = 16
cfg.NUM_SEGMENTS = cfg.RAW_WINDOW_SIZE // cfg.DOWNSAMPLE_RATE

cfg.UP_SCALE = 1

              
cfg.BATCH_SIZE = 32
cfg.LR_BASE = 1e-4
cfg.LR_STEP_SIZE = 10
cfg.LR_GAMMA = 0.9
cfg.NUM_EPOCHS = 80
cfg.TRAIN_BACKBONE = False

              
cfg.DATASET_DIR = os.environ.get("HANGTIME_DATA_DIR", os.path.join(os.getcwd(), "data", "HANGTIME"))
cfg.PRETRAIN_DIR = os.environ.get("HANGTIME_PRETRAIN_DIR", os.path.join(os.getcwd(), "pre_train"))

                 
cfg.LAMBDA = 0.01
cfg.R_EASY = 5
cfg.R_HARD = 10
cfg.m = 1
cfg.M = 10
cfg.CLASS_THRESH = 0
cfg.NMS_THRESH = 0.4
cfg.CAS_THRESH = np.arange(0.1, 0.5, 0.05)
cfg.ANESS_THRESH = np.arange(0.1, 0.9, 0.05)
cfg.TIOU_THRESH = np.linspace(0.3, 0.7, 5)

cfg.CLASS_DICT = {
        "dribbling": 0,
        "shot": 1,
        "pass": 2,
        "rebound": 3,
        "layup": 4
    }
