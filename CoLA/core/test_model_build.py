import torch
import sys
import os
from easydict import EasyDict as edict

# 引入你的模型代码
sys.path.append(os.getcwd())
from core.model import CoLA


def check_model():
    print("======== 开始测试模型结构 ========")

    # 1. 模拟 Config (与 core/config_xrfv2.py 保持一致)
    cfg = edict()
    cfg.FEATS_DIM = 30  # 输入维度 (IMU)
    cfg.NUM_CLASSES = 30  # 类别数
    cfg.R_EASY = 5
    cfg.R_HARD = 20
    cfg.m = 3
    cfg.M = 6

    # 2. 构造模型
    try:
        model = CoLA(cfg).cuda()
        print("✅ 模型实例化成功")
        print(f"   Backbone Input Dim: {model.actionness_module.len_feature}")
    except Exception as e:
        print(f"❌ 模型实例化失败: {e}")
        return

    # 3. 构造伪数据 (模拟 DataLoader 输出)
    # Shape: [Batch, Segments, Channels] -> [4, 2048, 30]
    dummy_input = torch.randn(4, 2048, 30).cuda()
    print(f"   Input Shape: {dummy_input.shape}")

    # 4. 前向传播测试
    try:
        # Forward
        video_scores, contrast_pairs, actionness, cas = model(dummy_input)

        # 检查输出
        print("\n✅ 前向传播成功! 输出检查:")
        print(f"   - Video Scores: {video_scores.shape} (Expect: [4, 30])")
        print(f"   - Actionness:   {actionness.shape}   (Expect: [4, 2048])")
        print(f"   - CAS (Class Activation): {cas.shape} (Expect: [4, 2048, 30])")

        # 检查对比学习对 (Contrast Pairs)
        print("   - Contrast Pairs:")
        for key, val in contrast_pairs.items():
            # Shape 应该是 [4, k, 2048]
            print(f"     * {key}: {val.shape}")

    except RuntimeError as e:
        print(f"\n❌ 前向传播报错: {e}")
        print("提示: 请检查 model.py 中 permute 的维度顺序")
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")


if __name__ == '__main__':
    check_model()