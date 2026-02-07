import torch
from types import SimpleNamespace
from RSKP_MODEL.main_branch import WSTAL
import argparse


def calculate_parameters(model):
    """computemodelparameterscount"""
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{name}: {param_count} parameters")
    print(f"\nTotal trainable parameters: {total_params}")
    print(f"Total trainable parameters (in millions): {total_params / 1e6:.2f}M")
    return total_params


def main():
    parser = argparse.ArgumentParser(description="Calculate model parameters from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    print("Loaded checkpoint successfully!")
    print(f"Checkpoint keys: {list(ckpt.keys())}")

    args_dict = ckpt.get("args", {})
    num_classes = ckpt.get("num_classes", 5)

    print(f"\nModel configuration:")
    print(f"args: {args_dict}")
    print(f"num_classes: {num_classes}")

    model_args = SimpleNamespace(
        w=args_dict.get("w", 0.5),
        inp_feat_num=args_dict.get("inp_feat_num", 512),
        out_feat_num=args_dict.get("out_feat_num", 512),
        mu_num=args_dict.get("mu_num", 8),
        em_iter=args_dict.get("em_iter", 2),
        class_num=num_classes,
        scale_factor=args_dict.get("scale_factor", 20.0),
        dropout=args_dict.get("dropout", 0.6),
        mu_queue_len=args_dict.get("mu_queue_len", 5),
    )

    model = WSTAL(model_args)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("\nLoaded model state dict successfully!")
    else:
        model.load_state_dict(ckpt)
        print("\nLoaded entire checkpoint as model state dict!")

    print("\nModel parameters:")
    calculate_parameters(model)


if __name__ == "__main__":
    main()
