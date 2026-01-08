# test_wscnet_imu.py

import os
import json
import time
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
from builder_models import build_wsddn_imu_model, build_pcl_oicr_imu_model
from models.WSDDN_model import generate_proposal_boxes
from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTest, FullBackboneWrapper1D
from tool import load_label_mapping, softnms_v2, ANETdetection
from builder_pretrainbackbone import load_pretrained_backbone



def test_wsddn_imu(config, checkpoint_path,test_mode: str = "test_window"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n开始测试，测试设备：{device}")
    print(f"加载模型：{checkpoint_path}")

    # 加载checkpoint，拿到训练时的 use_airpods / in_channels / pretrained_name ===
    checkpoint = torch.load(checkpoint_path, map_location=device)
    use_airpods = bool(checkpoint.get("use_airpods", config["training"].get("use_airpods", False)))
    in_channels = int(checkpoint.get("in_channels", 30 + (6 if use_airpods else 0)))
    pretrained_name = checkpoint.get("pretrained_name", config["model"]["pretrained_name"])

    print(f"[Test] use_airpods = {use_airpods}, in_channels = {in_channels}")
    print(f"[Test] pretrained_name = {pretrained_name}")
    print(f"[Test] has backbone_state_dict = {'backbone_state_dict' in checkpoint}")

    # 1. 初始化测试数据集
    test_dataset = WeaklySupervisedXRFV2DatasetTest(
        config=config,
        modality='imu',
        device_keep_list=config["testing"]["device_keep_list"],
        use_airpods=use_airpods,
    )


    # 2. 构建 backbone（先按pretrained_name构建+尝试加载预训练；如果ckpt里有训练后权重则覆盖）
    pretrained_backbone, T_global = load_pretrained_backbone(
        pretrained_name=config["model"]["pretrained_name"],
        device=device,
        in_channels=in_channels,
    )
    pretrained_backbone = pretrained_backbone.to(device)
    # 若checkpoint中保存了训练后的backbone权重，则优先使用它 ===
    if "backbone_state_dict" in checkpoint:
        missing, unexpected = pretrained_backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=False)
        print("[Test] 使用训练后的backbone权重（来自checkpoint）")
        print(f"  missing={missing}")
        print(f"  unexpected={unexpected}")
    else:
        print("[Test] 使用预训练backbone权重（来自PRETRAINED_ZOO ckpt）")

    pretrained_backbone.eval()

    # full 模式：用 wrapper 把 backbone 扩展到整条序列
    if test_mode == "test_full":
        full_win_len = int(config["testing"].get("full_win_len", test_dataset.clip_length))  # 默认=2048
        full_stride = int(config["testing"].get("full_stride", test_dataset.stride))  # 默认=window stride
        full_wrapper = None
        full_wrapper = FullBackboneWrapper1D(pretrained_backbone, win_len=full_win_len, stride=full_stride,
                                             in_channels=in_channels).to(device)
        full_wrapper.eval()
        print(f"[Test-Full] FullBackboneWrapper: win_len={full_win_len}, stride={full_stride}")

    # 3. 标签映射 & 模型构建
    _, _, new_to_action = load_label_mapping(config["path"]["mapping_path"])
    num_classes = len(new_to_action)
    model = build_wsddn_imu_model(config, num_classes, device)

    # 加载head最优权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # 兼容直接保存的state_dict
    model.eval()

    # 3. 测试参数配置
    conf_thresh = config['testing']['conf_thresh']
    nms_sigma = config['testing']['nms_sigma']
    top_k = config['testing']['top_k']
    # num_proposals = config['testing']['num_proposals']
    num_proposals = int(config["testing"].get(
        "num_proposals_full" if test_mode == "test_full" else "num_proposals_window",800))
    result_path = config["path"]["result_path"]
    os.makedirs(result_path, exist_ok=True)

    # 4. 获取全局特征时序长度T_global
    dummy_input = torch.randn(2, in_channels, 2048, device=device)
    with torch.no_grad():
        global_feat = pretrained_backbone(dummy_input)
    T_global = global_feat.shape[2]
    print(f"全局特征时序长度T_global：{T_global}")

    # 5. 生成测试候选框
    proposal_boxes = generate_proposal_boxes(
        T_global=T_global,
        num_proposals=num_proposals
    ).to(device)

    # 6. 开始批量推理
    result_dict = {}
    inf_time_list = []
    gpu_mem_list = []

    test_files = list(test_dataset.dataset())
    num_test_files = len(test_files)

    for file_name, data_iterator in tqdm(test_files, desc="测试进度", unit="文件", total=num_test_files):
        class_outputs = [[] for _ in range(num_classes)]
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # ========== A) test_full：整条序列只跑一次 head ==========
        if test_mode == "test_full":
            # data_iterator 只会 yield 1 次：({'imu': [C,T_total]}, [0,T_total])
            clip, segment = next(iter(data_iterator))
            imu_full = clip['imu'].to(device).unsqueeze(0)  # [1,C,T_total]
            win_start, win_end = segment
            win_len = int(win_end - win_start)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            st = time.time()

            with torch.no_grad():
                global_feat, info = full_wrapper(imu_full, return_info=True)  # [1,D,T_global_full]
                T_global = global_feat.shape[2]
                proposal_boxes = generate_proposal_boxes(T_global=T_global, num_proposals=num_proposals).to(device)
                outputs = model(global_feat, proposal_boxes.unsqueeze(0))

            if device.type == 'cuda':
                torch.cuda.synchronize()
                gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
            inf_time_list.append((time.time() - st) * 1000)

            joint_prob = outputs["joint_prob"][0]  # [P,C]
            for p in range(proposal_boxes.shape[0]):
                s_idx = float(proposal_boxes[p, 0].item())
                e_idx = float(proposal_boxes[p, 1].item())

                # ★映射回 raw：用 win_len，不用 test_dataset.clip_length
                raw_start = win_start + int(s_idx / T_global * win_len)
                raw_end = win_start + int(e_idx / T_global * win_len)
                raw_end = min(raw_end, win_end)

                for cl in range(num_classes):
                    score = float(joint_prob[p, cl].item())
                    if score >= conf_thresh:
                        class_outputs[cl].append([raw_start, raw_end, score])

        # ========== B) test_window：（但修正映射公式） ==========
        else:
            for clip, segment in data_iterator:
                imu_clip = clip['imu'].to(device).unsqueeze(0)  # [1,C,2048]
                win_start, win_end = segment
                win_len = int(win_end - win_start)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                st = time.time()

                with torch.no_grad():
                    global_feat = pretrained_backbone(imu_clip)
                    T_global = global_feat.shape[2]
                    proposal_boxes = generate_proposal_boxes(T_global=T_global, num_proposals=num_proposals).to(
                        device)
                    outputs = model(global_feat, proposal_boxes.unsqueeze(0))

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                inf_time_list.append((time.time() - st) * 1000)

                joint_prob = outputs["joint_prob"][0]  # [P,C]
                for p in range(proposal_boxes.shape[0]):
                    s_idx = float(proposal_boxes[p, 0].item())
                    e_idx = float(proposal_boxes[p, 1].item())

                    # ★统一修正映射：按该 window 的真实长度映射
                    raw_start = win_start + int(s_idx / T_global * win_len)
                    raw_end = win_start + int(e_idx / T_global * win_len)
                    raw_end = min(raw_end, win_end)

                    for cl in range(num_classes):
                        score = float(joint_prob[p, cl].item())
                        if score >= conf_thresh:
                            class_outputs[cl].append([raw_start, raw_end, score])

            if device.type == 'cuda':
                gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

        # 每个类别做 Soft-NMS
        final_proposals = []
        for cl in range(num_classes):
            if not class_outputs[cl]:
                # print(f"file_name：{file_name}的{cl}类别无有效候选，置信度都低于阈值")
                continue

            segments = torch.tensor(class_outputs[cl], dtype=torch.float32)
            segments_nms, _ = softnms_v2(
                segments,
                sigma=nms_sigma,
                top_k=top_k,
                score_threshold=conf_thresh
            )
            class_name = test_dataset.id_to_action.get(str(cl), f"class_{cl}")
            for seg in segments_nms:
                final_proposals.append({
                    "label": class_name,
                    "score": float(seg[2]),
                    "segment": [float(seg[0]), float(seg[1])]
                })

        result_dict[file_name] = final_proposals

    # 7. 保存推理性能统计
    avg_inf_time = np.mean(inf_time_list) if inf_time_list else 0.0
    std_inf_time = np.std(inf_time_list) if inf_time_list else 0.0
    avg_gpu_mem = np.mean(gpu_mem_list) if gpu_mem_list else 0.0
    std_gpu_mem = np.std(gpu_mem_list) if gpu_mem_list else 0.0

    stats_data = {
        "测试文件总数": num_test_files,
        "平均窗口推理时间(ms)": f"{avg_inf_time:.2f} ± {std_inf_time:.2f}",
        "平均GPU峰值内存(MB)": f"{avg_gpu_mem:.2f} ± {std_gpu_mem:.2f}",
        "置信度阈值": conf_thresh,
        "Soft-NMS sigma": nms_sigma,
        "测试候选数": num_proposals
    }
    with open(os.path.join(result_path, "inference_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    # 8. 保存预测结果
    prediction_data = {
        "version": "WSDDN-IMU-v1.0",
        "results": result_dict,
        "external_data": {}
    }
    pred_save_path = os.path.join(result_path, f"predictions_{test_mode}.json")
    with open(pred_save_path, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, indent=2, ensure_ascii=False)
    print(f"预测结果已保存至：{pred_save_path}")

    # 9. 评估 mAP
    print("\n开始评估动作定位性能...")
    tious = np.linspace(0.3, 0.7, 5)
    anet_evaluator = ANETdetection(
        ground_truth_filename=test_dataset.eval_gt,
        prediction_filename=pred_save_path,
        subset='test',
        tiou_thresholds=tious
    )
    mAPs, avg_mAP, _ = anet_evaluator.evaluate()
    print(f"[ANET] {test_mode} avg_mAP={avg_mAP:.4f}")

    # 10. 报告
    report_content = [
        "=" * 60,
        "WSDDN-IMU 训练+测试综合报告",
        "=" * 60,
        f"训练配置：",
        f"  - 训练数据集：{config['path']['train_dataset_path']}",
        f"  - 训练轮数：{config['training']['num_epochs']}",
        f"  - 批次大小：{config['training']['batch_size']}",
        f"  - 训练候选数：{config['training']['num_proposals']}",
        f"  - 最优模型路径：{checkpoint_path}",
        f"",
        f"测试配置：",
        f"  - 测试数据集：{config['path']['test_dataset_path']}",
        f"  - 测试候选数：{num_proposals}",
        f"  - 置信度阈值：{conf_thresh}",
        f"  - Soft-NMS sigma：{nms_sigma}",
        f"  - 测试文件总数：{num_test_files}",
        f"",
        f"推理性能：",
        f"  - 平均窗口推理时间：{avg_inf_time:.2f} ± {std_inf_time:.2f} ms",
        f"  - 平均GPU峰值内存：{avg_gpu_mem:.2f} ± {std_gpu_mem:.2f} MB",
        f"",
        f"动作定位评估结果（tIoU=0.3~0.7）：",
        "-" * 60
    ]
    for tiou, mAP in zip(tious, mAPs):
        report_content.append(f"tIoU={tiou:.2f} → mAP={mAP:.4f}")
    report_content.extend([
        "-" * 60,
        f"平均mAP：{avg_mAP:.4f}",
        "=" * 60
    ])

    report_save_path = os.path.join(result_path, f"train_{test_mode}_report.txt")
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    print("\n" + "\n".join(report_content))
    print(f"\n测试完成！所有结果已保存至：{result_path}")


def test_pcl_imu(config, checkpoint_path,test_mode: str = "test_window"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[PCL/OICR] 开始测试，测试设备：{device}")
    print(f"[PCL/OICR] 加载模型：{checkpoint_path}")

    # 先加载checkpoint，拿到训练时的 use_airpods/in_channels/pretrained_name ===
    checkpoint = torch.load(checkpoint_path, map_location=device)
    use_airpods = bool(checkpoint.get("use_airpods", config["training"].get("use_airpods", False)))
    in_channels = int(checkpoint.get("in_channels", 30 + (6 if use_airpods else 0)))
    pretrained_name = checkpoint.get("pretrained_name", config["model"]["pretrained_name"])

    print(f"[PCL/OICR-Test] use_airpods = {use_airpods}, in_channels = {in_channels}")
    print(f"[PCL/OICR-Test] pretrained_name = {pretrained_name}")
    print(f"[PCL/OICR-Test] has backbone_state_dict = {'backbone_state_dict' in checkpoint}")

    # 1) 测试数据集
    test_dataset = WeaklySupervisedXRFV2DatasetTest(
        config=config,
        modality='imu',
        device_keep_list=config['testing'].get('device_keep_list', None),
        use_airpods=use_airpods,
    )

    # 2) 加载 pretrained_backbone
    pretrained_backbone, T_global = load_pretrained_backbone(
        pretrained_name=config["model"]["pretrained_name"],
        device=device,
        in_channels=in_channels,
    )
    pretrained_backbone = pretrained_backbone.to(device)
    # 若checkpoint中保存了训练后的backbone权重，则优先使用它 ===
    if "backbone_state_dict" in checkpoint:
        missing, unexpected = pretrained_backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=False)
        print("[PCL/OICR-Test] 使用训练后的backbone权重（来自checkpoint）")
        print(f"  missing={missing}")
        print(f"  unexpected={unexpected}")
    else:
        print("[PCL/OICR-Test] 使用预训练backbone权重（来自PRETRAINED_ZOO ckpt）")

    pretrained_backbone.eval()

    # full 模式：用 wrapper 把 backbone 扩展到整条序列
    if test_mode == "test_full":
        full_win_len = int(config["testing"].get("full_win_len", test_dataset.clip_length))  # 默认=2048
        full_stride = int(config["testing"].get("full_stride", test_dataset.stride))  # 默认=window stride
        full_wrapper = None
        full_wrapper = FullBackboneWrapper1D(pretrained_backbone, win_len=full_win_len, stride=full_stride,
                                             in_channels=in_channels).to(device)
        full_wrapper.eval()
        print(f"[Test-Full] FullBackboneWrapper: win_len={full_win_len}, stride={full_stride}")

    # 3) 模型
    _, _, new_to_action = load_label_mapping(config["path"]["mapping_path"])
    num_classes = len(new_to_action)
    model = build_pcl_oicr_imu_model(config, num_classes, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 4) 测试参数
    conf_thresh = config['testing']['conf_thresh']
    nms_sigma = config['testing']['nms_sigma']
    top_k = config['testing']['top_k']
    # num_proposals = config['testing']['num_proposals']
    num_proposals = int(config["testing"].get(
        "num_proposals_full" if test_mode == "test_full" else "num_proposals_window",800))

    result_path = config["path"]["result_path"]
    os.makedirs(result_path, exist_ok=True)

    print(f"[PCL/OICR] 全局特征时序长度 T_global：{T_global}")

    proposal_boxes = generate_proposal_boxes(
        T_global=T_global,
        num_proposals=num_proposals
    ).to(device)  # [P,2]

    # 5) 推理
    result_dict = {}
    inf_time_list, gpu_mem_list = [], []

    test_files = list(test_dataset.dataset())
    num_test_files = len(test_files)

    for file_name, data_iterator in tqdm(test_files, desc="[PCL/OICR] 测试进度", unit="文件", total=num_test_files):
        class_outputs = [[] for _ in range(num_classes)]

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # ========== A) test_full：整条序列只跑一次 head ==========
        if test_mode == "test_full":
            # data_iterator 只会 yield 1 次：({'imu': [C,T_total]}, [0,T_total])
            clip, segment = next(iter(data_iterator))
            imu_full = clip['imu'].to(device).unsqueeze(0)  # [1,C,T_total]
            win_start, win_end = segment
            win_len = int(win_end - win_start)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            st = time.time()

            with torch.no_grad():
                global_feat, info = full_wrapper(imu_full, return_info=True)  # [1,D,T_global_full]
                T_global = global_feat.shape[2]
                proposal_boxes = generate_proposal_boxes(T_global=T_global, num_proposals=num_proposals).to(device)
                outputs = model(global_feat, proposal_boxes.unsqueeze(0))

            if device.type == 'cuda':
                torch.cuda.synchronize()
                gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
            inf_time_list.append((time.time() - st) * 1000)

            joint_prob = outputs["joint_prob"][0]  # [P,C]
            for p in range(proposal_boxes.shape[0]):
                s_idx = float(proposal_boxes[p, 0].item())
                e_idx = float(proposal_boxes[p, 1].item())

                # ★映射回 raw：用 win_len，不用 test_dataset.clip_length
                raw_start = win_start + int(s_idx / T_global * win_len)
                raw_end = win_start + int(e_idx / T_global * win_len)
                raw_end = min(raw_end, win_end)

                for cl in range(num_classes):
                    score = float(joint_prob[p, cl].item())
                    if score >= conf_thresh:
                        class_outputs[cl].append([raw_start, raw_end, score])

        # ========== B) test_window：（但修正映射公式） ==========
        else:
            for clip, segment in data_iterator:
                imu_clip = clip['imu'].to(device).unsqueeze(0)  # [1,C,2048]
                win_start, win_end = segment
                win_len = int(win_end - win_start)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                st = time.time()

                with torch.no_grad():
                    global_feat = pretrained_backbone(imu_clip)
                    T_global = global_feat.shape[2]
                    proposal_boxes = generate_proposal_boxes(T_global=T_global, num_proposals=num_proposals).to(
                        device)
                    outputs = model(global_feat, proposal_boxes.unsqueeze(0))

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                inf_time_list.append((time.time() - st) * 1000)

                joint_prob = outputs["joint_prob"][0]  # [P,C]
                for p in range(proposal_boxes.shape[0]):
                    s_idx = float(proposal_boxes[p, 0].item())
                    e_idx = float(proposal_boxes[p, 1].item())

                    # ★统一修正映射：按该 window 的真实长度映射
                    raw_start = win_start + int(s_idx / T_global * win_len)
                    raw_end = win_start + int(e_idx / T_global * win_len)
                    raw_end = min(raw_end, win_end)

                    for cl in range(num_classes):
                        score = float(joint_prob[p, cl].item())
                        if score >= conf_thresh:
                            class_outputs[cl].append([raw_start, raw_end, score])

            if device.type == 'cuda':
                gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

        # Soft-NMS
        final_proposals = []
        for cl in range(num_classes):
            if not class_outputs[cl]:
                continue
            segments = torch.tensor(class_outputs[cl], dtype=torch.float32)
            segments_nms, _ = softnms_v2(
                segments,
                sigma=nms_sigma,
                top_k=top_k,
                score_threshold=conf_thresh
            )
            class_name = test_dataset.id_to_action.get(str(cl), f"class_{cl}")
            for seg in segments_nms:
                final_proposals.append({
                    "label": class_name,
                    "score": float(seg[2]),
                    "segment": [float(seg[0]), float(seg[1])]
                })

        result_dict[file_name] = final_proposals

    # 6) 统计
    avg_inf_time = np.mean(inf_time_list) if inf_time_list else 0.0
    std_inf_time = np.std(inf_time_list) if inf_time_list else 0.0
    avg_gpu_mem = np.mean(gpu_mem_list) if gpu_mem_list else 0.0
    std_gpu_mem = np.std(gpu_mem_list) if gpu_mem_list else 0.0

    stats_data = {
        "测试文件总数": num_test_files,
        "平均窗口推理时间(ms)": f"{avg_inf_time:.2f} ± {std_inf_time:.2f}",
        "平均GPU峰值内存(MB)": f"{avg_gpu_mem:.2f} ± {std_gpu_mem:.2f}",
        "use_airpods": use_airpods,
        "in_channels": in_channels,
        "置信度阈值": conf_thresh,
        "Soft-NMS sigma": nms_sigma,
        "测试候选数": num_proposals
    }
    with open(os.path.join(result_path, "inference_stats_pcl.json"), "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    prediction_data = {
        "version": "PCL-OICR-IMU-v1.0",
        "results": result_dict,
        "external_data": {}
    }
    pred_save_path = os.path.join(result_path, f"predictions_{test_mode}.json")
    with open(pred_save_path, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, indent=2, ensure_ascii=False)
    print(f"[PCL/OICR] 预测结果已保存至：{pred_save_path}")

    # 7) mAP
    print("\n[PCL/OICR] 开始评估动作定位性能...")
    tious = np.linspace(0.3, 0.7, 5)
    anet_evaluator = ANETdetection(
        ground_truth_filename=test_dataset.eval_gt,
        prediction_filename=pred_save_path,
        subset='test',
        tiou_thresholds=tious
    )
    mAPs, avg_mAP, _ = anet_evaluator.evaluate()

    report_content = [
        "=" * 60,
        "PCL/OICR-IMU 训练+测试综合报告",
        "=" * 60,
        f"测试配置：use_airpods={use_airpods}, in_channels={in_channels}",
        f"测试文件总数：{num_test_files}",
        "-" * 60,
    ]
    for tiou, mAP in zip(tious, mAPs):
        report_content.append(f"tIoU={tiou:.2f} → mAP={mAP:.4f}")
    report_content.append(f"平均mAP：{avg_mAP:.4f}")
    report_content.append("=" * 60)

    report_save_path = os.path.join(result_path, f"train_{test_mode}_report_pcl.txt")
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    print("\n" + "\n".join(report_content))
    print(f"\n[PCL/OICR] 测试完成！所有结果已保存至：{result_path}")

