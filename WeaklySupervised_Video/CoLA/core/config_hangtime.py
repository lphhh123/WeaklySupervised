import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()

# --- 基础配置 ---
cfg.GPU_ID = '0'
cfg.SEED = 2024
cfg.NUM_WORKERS = 4

# --- 数据集特定配置 (HANGTIME) ---
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

# --- 训练配置 ---
cfg.BATCH_SIZE = 32
cfg.LR_BASE = 1e-4
cfg.LR_STEP_SIZE = 10
cfg.LR_GAMMA = 0.9
cfg.NUM_EPOCHS = 80
cfg.TRAIN_BACKBONE = False

# --- 路径配置 ---
cfg.DATASET_DIR = "/home/lipei/TAL_data/hangtime/" # 数据根目录
cfg.PRETRAIN_DIR = "/home/lipei/project/WSDDN/OtherData/HANGTIME/pre_train/CNN1D" # 预训练权重目录

# --- CoLA 参数 ---
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