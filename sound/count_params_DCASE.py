import torch
import torch.nn as nn
from models.DCASE_CRNN import CRNN


def count_parameters_manually(model):
    print(f"{'Module':<20} | {'Layer Name':<45} | {'Parameters':<10}")
    print("-" * 80)

    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            p_count = param.numel()
            print(f"{name.split('.')[0]:<20} | {name:<45} | {p_count:>10,}")
            total_params += p_count

    print("-" * 80)
    print(f"总可训练参数量 (Total Trainable Params): {total_params:,}")
    return total_params


if __name__ == "__main__":


    model = CRNN(n_in_channel=1, nclass=10, activation="glu")

    print("\n[DCASE CRNN 参数量手动统计]")
    count_parameters_manually(model)