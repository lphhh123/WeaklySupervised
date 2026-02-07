import os
import torch
import glob
import numpy as np
from terminaltables import AsciiTable



PATHS = {

    "Opportunity": "./output_opportunity_cola_ddp",
    "Hangtime": "./output_hangtime_cola_ddp",
    "RWHAR": "./output_rwhar_cola_ddp",
    "SBHAR": "./output_sbhar_cola_ddp",
    "WEAR": "./output_wear_cola_ddp",
    "WETLAB": "./output_wetlab_cola_ddp",


    "XRFV2": "./output_ddp"
}


FILENAME_PATTERNS = ["model_final.pth", "model_best.pth"]


# ===========================================

def count_parameters_in_ckpt(ckpt_path):
    """Message"""
    try:

        checkpoint = torch.load(ckpt_path, map_location="cpu")



        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:

                state_dict = checkpoint
        else:
            print(f"   ❌ Message: {ckpt_path}")
            return None


        total_params = 0
        for k, v in state_dict.items():

            if "num_batches_tracked" in k:
                continue
            if v.dim() == 0:
                continue
            total_params += v.numel()

        return total_params

    except Exception as e:
        print(f"   ❌ Message {ckpt_path}: {e}")
        return None


def get_files_standard(root_dir):
    """
    Message: root/seed_*/fold*/model_final.pth
    """
    files = []

    for pattern in FILENAME_PATTERNS:

        search_path = os.path.join(root_dir, "seed_*", "fold*", pattern)
        found = glob.glob(search_path)
        files.extend(found)
    return files


def get_files_xrfv2(root_dir):
    """
    XRFV2 Message: root/seed_*/checkpoints/model_best.pth
    """
    files = []
    for pattern in FILENAME_PATTERNS:

        search_path = os.path.join(root_dir, "cnn1d_*", "checkpoints", pattern)
        found = glob.glob(search_path)
        files.extend(found)
    return files


def main():
    print("🚀 Message...")
    results = []
    headers = ["Dataset", "Files Found", "Avg Params (M)", "Bytes (MB)"]

    for name, root_path in PATHS.items():
        if not os.path.exists(root_path):
            results.append([name, "0", "N/A", "N/A"])
            continue


        if name == "XRFV2":
            files = get_files_xrfv2(root_path)
        else:
            files = get_files_standard(root_path)

        if not files:
            results.append([name, "0", "N/A", "N/A"])
            continue



        sample_files = files[:3]
        counts = []

        for f in sample_files:
            c = count_parameters_in_ckpt(f)
            if c is not None:
                counts.append(c)

        if not counts:
            results.append([name, str(len(files)), "Error", "Error"])
            continue


        avg_count = np.mean(counts)
        avg_million = avg_count / 1e6

        size_mb = (avg_count * 4) / (1024 * 1024)

        results.append([
            name,
            str(len(files)),
            f"{avg_million:.2f} M",
            f"{size_mb:.2f} MB"
        ])


    table = AsciiTable([headers] + results)
    print("\n" + table.table)
    print("\nMessage: Message(M), Message float32 Message。")


if __name__ == "__main__":
    main()