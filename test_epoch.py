# test_wscnet_imu.py
import copy
import os
import json
import time
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
from builder_models import build_wscnet_imu_model, build_wsddn_imu_model, build_pcl_oicr_imu_model, \
    build_pclHead_model, build_oicrPALoss_imu_model
from metrics_utils import evaluate_extra_metrics_and_plots
from models.WSCNet_model import fuse_wscnet_outputs
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
        mode=test_mode,
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
        full_wrapper = FullBackboneWrapper1D(pretrained_backbone, win_len=full_win_len, stride=full_stride,in_channels=in_channels).to(device)
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

        # for clip, segment in data_iterator:
        #     imu_clip = clip['imu'].to(device).unsqueeze(0)
        #     B = imu_clip.shape[0]
        #
        #     if device.type == 'cuda':
        #         torch.cuda.synchronize()
        #     start_time = time.time()
        #
        #     with torch.no_grad():
        #         global_feat = pretrained_backbone(imu_clip)
        #         batch_proposals = proposal_boxes.unsqueeze(0).repeat(B, 1, 1)
        #         outputs = model(global_feat, batch_proposals)
        #
        #     if device.type == 'cuda':
        #         torch.cuda.synchronize()
        #     inf_time = (time.time() - start_time) * 1000
        #     inf_time_list.append(inf_time)
        #
        #     joint_prob = outputs["joint_prob"][0]  # [P, C]
        #
        #     # 映射候选框到原始数据的时间位置
        #     for p in range(num_proposals):
        #         start_idx, end_idx = proposal_boxes[p]
        #         feat_len = end_idx - start_idx
        #         raw_len = int(feat_len / T_global * test_dataset.clip_length)
        #
        #         window_start, window_end = segment
        #         raw_start = window_start + int(start_idx / T_global * test_dataset.clip_length)
        #         raw_end = raw_start + raw_len
        #         raw_end = min(raw_end, window_end)
        #
        #         for cl in range(num_classes):
        #             score = joint_prob[p, cl].item()
        #             if score > conf_thresh:
        #                 class_outputs[cl].append([raw_start, raw_end, score])
        #
        #     if device.type == 'cuda':
        #         peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        #         gpu_mem_list.append(peak_mem)

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

    # ===== 指标计算：PRF + confusion + UR/OR/DR/IR/FR/MR =====
    class_names = [test_dataset.id_to_action.get(str(i), f"class_{i}") for i in range(num_classes)]
    conf_thr = float(config.get("testing", {}).get("confusion_tiou", 0.5))

    extra_lines, artifacts = evaluate_extra_metrics_and_plots(
        gt_path=test_dataset.eval_gt,
        pred_path=pred_save_path,
        class_names=class_names,
        tious=tious,
        result_dir=result_path,
        test_mode=test_mode,
        subset_name="test",
        confusion_tiou=conf_thr,
        include_bg=True,
        uodifm_unit="frame",  # 按帧计算
        seconds_per_unit=None,  # frame模式不需要
        cm_value_thresh=0.02,
    )

    report_content.extend(extra_lines)

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
        mode=test_mode,
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

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # 4) 测试参数
    conf_thresh = config['testing']['conf_thresh']
    nms_sigma = config['testing']['nms_sigma']
    top_k = config['testing']['top_k']
    # num_proposals = config['testing']['num_proposals']
    num_proposals = int(config["testing"].get(
        "num_proposals_full" if test_mode == "test_full" else "num_proposals_window", 800))
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
    print(f"[ANET] {test_mode} avg_mAP={avg_mAP:.4f}")

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

    # ===== 指标计算：PRF + confusion + UR/OR/DR/IR/FR/MR =====
    class_names = [test_dataset.id_to_action.get(str(i), f"class_{i}") for i in range(num_classes)]
    conf_thr = float(config.get("testing", {}).get("confusion_tiou", 0.5))

    extra_lines, artifacts = evaluate_extra_metrics_and_plots(
        gt_path=test_dataset.eval_gt,
        pred_path=pred_save_path,
        class_names=class_names,
        tious=tious,
        result_dir=result_path,
        test_mode=test_mode,
        subset_name="test",
        confusion_tiou=conf_thr,
        include_bg=True,
        uodifm_unit="frame",  # 按帧计算
        seconds_per_unit=None,  # frame模式不需要
        cm_value_thresh=0.02,
    )

    report_content.extend(extra_lines)

    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    print("\n" + "\n".join(report_content))
    print(f"\n[PCL/OICR] 测试完成！所有结果已保存至：{result_path}")


def test_oicrPALoss_imu(config, checkpoint_path, test_mode: str = "test_window"):
    import os, json, time
    import numpy as np
    import torch
    from tqdm import tqdm

    assert test_mode in ["test_window", "test_full"]

    def _extract_joint_prob(outputs: dict, num_classes: int) -> torch.Tensor:
        """
        返回 [B,P,C] 的 joint_prob.
        IMU_OICR_PALoss 应该直接有 joint_prob；这里留个保险 fallback。
        """
        jp = outputs.get("joint_prob", None)
        if jp is not None:
            if jp.dim() == 2:
                jp = jp.unsqueeze(0)
            return jp

        refine_scores = outputs.get("refine_scores", None)
        if refine_scores is None:
            raise KeyError("outputs must contain 'joint_prob' or 'refine_scores' in eval.")
        acc = None
        for rs in refine_scores:
            rs_fg = rs[..., 1:] if rs.size(-1) == num_classes + 1 else rs
            acc = rs_fg if acc is None else (acc + rs_fg)
        return acc / float(len(refine_scores))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[OICR-PA] 开始测试，device={device}")
    print(f"[OICR-PA] ckpt={checkpoint_path}, mode={test_mode}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    use_airpods = bool(checkpoint.get("use_airpods", config["training"].get("use_airpods", False)))
    in_channels = int(checkpoint.get("in_channels", 30 + (6 if use_airpods else 0)))
    pretrained_name = checkpoint.get("pretrained_name", config["model"]["pretrained_name"])

    print(f"[OICR-PA] use_airpods={use_airpods}, in_channels={in_channels}")
    print(f"[OICR-PA] pretrained_name={pretrained_name}")
    print(f"[OICR-PA] has backbone_state_dict={'backbone_state_dict' in checkpoint}")

    # 复制一份 config，确保 builder 用到的 pretrained_name 一致
    config = copy.deepcopy(config)
    config["model"]["pretrained_name"] = pretrained_name
    config["training"]["use_airpods"] = use_airpods

    # 1) dataset
    test_dataset = WeaklySupervisedXRFV2DatasetTest(
        config=config,
        modality="imu",
        device_keep_list=config["testing"].get("device_keep_list", None),
        use_airpods=use_airpods,
        mode=test_mode,
    )

    # 2) backbone
    pretrained_backbone, _Tg = load_pretrained_backbone(
        pretrained_name=pretrained_name,
        device=device,
        in_channels=in_channels,
    )
    pretrained_backbone = pretrained_backbone.to(device)

    if "backbone_state_dict" in checkpoint:
        # 训练后权重优先；优先 strict=True
        try:
            pretrained_backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
            print("[OICR-PA] 使用训练后的 backbone（strict=True）")
        except Exception:
            missing, unexpected = pretrained_backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=False)
            print("[Warn] strict=True failed, fallback strict=False")
            print(f"  missing={missing}")
            print(f"  unexpected={unexpected}")
    else:
        print("[OICR-PA] 使用 PRETRAINED_ZOO backbone（无训练后覆盖）")

    pretrained_backbone.eval()

    # full 模式 wrapper
    full_wrapper = None
    if test_mode == "test_full":
        full_win_len = int(config["testing"].get("full_win_len", test_dataset.clip_length))
        full_stride = int(config["testing"].get("full_stride", test_dataset.stride))
        full_wrapper = FullBackboneWrapper1D(
            pretrained_backbone, win_len=full_win_len, stride=full_stride, in_channels=in_channels
        ).to(device)
        full_wrapper.eval()
        print(f"[OICR-PA] FullBackboneWrapper: win_len={full_win_len}, stride={full_stride}")

    # 3) head model（只构建 OICR-PA）
    _, _, new_to_action = load_label_mapping(config["path"]["mapping_path"])
    num_classes = len(new_to_action)

    model = build_oicrPALoss_imu_model(config, num_classes, device).to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # 4) test params
    conf_thresh = float(config["testing"]["conf_thresh"])
    nms_sigma = float(config["testing"]["nms_sigma"])
    top_k = int(config["testing"]["top_k"])

    num_proposals = int(config["testing"].get(
        "num_proposals_full" if test_mode == "test_full" else "num_proposals_window", 800
    ))
    result_path = config["path"]["result_path"]
    os.makedirs(result_path, exist_ok=True)

    # 5) inference loop
    result_dict = {}
    inf_time_list, gpu_mem_list = [], []

    test_files = list(test_dataset.dataset())
    num_test_files = len(test_files)

    for file_name, data_iterator in tqdm(test_files, desc=f"[OICR-PA] {test_mode}", unit="文件", total=num_test_files):
        class_outputs = [[] for _ in range(num_classes)]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if test_mode == "test_full":
            clip, segment = next(iter(data_iterator))
            imu_full = clip["imu"].to(device).unsqueeze(0)  # [1,C,T_total]
            win_start, win_end = segment
            win_len = int(win_end - win_start)

            if device.type == "cuda":
                torch.cuda.synchronize()
            st = time.time()

            with torch.no_grad():
                global_feat, _info = full_wrapper(imu_full, return_info=True)  # [1,D,Tg_full]
                T_global_now = int(global_feat.shape[2])
                prop = generate_proposal_boxes(T_global=T_global_now, num_proposals=num_proposals).to(device)
                outputs = model(global_feat, prop.unsqueeze(0))
                joint_prob = _extract_joint_prob(outputs, num_classes)[0]  # [P,C]

            if device.type == "cuda":
                torch.cuda.synchronize()
                gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
            inf_time_list.append((time.time() - st) * 1000)

            for p in range(prop.shape[0]):
                s_idx = float(prop[p, 0].item())
                e_idx = float(prop[p, 1].item())
                raw_start = win_start + int(s_idx / T_global_now * win_len)
                raw_end = win_start + int(e_idx / T_global_now * win_len)
                raw_end = min(raw_end, win_end)

                for cl in range(num_classes):
                    score = float(joint_prob[p, cl].item())
                    if score >= conf_thresh:
                        class_outputs[cl].append([raw_start, raw_end, score])

        else:
            for clip, segment in data_iterator:
                imu_clip = clip["imu"].to(device).unsqueeze(0)  # [1,C,2048]
                win_start, win_end = segment
                win_len = int(win_end - win_start)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                st = time.time()

                with torch.no_grad():
                    global_feat = pretrained_backbone(imu_clip)
                    T_global_now = int(global_feat.shape[2])
                    prop = generate_proposal_boxes(T_global=T_global_now, num_proposals=num_proposals).to(device)
                    outputs = model(global_feat, prop.unsqueeze(0))
                    joint_prob = _extract_joint_prob(outputs, num_classes)[0]  # [P,C]

                if device.type == "cuda":
                    torch.cuda.synchronize()
                inf_time_list.append((time.time() - st) * 1000)

                for p in range(prop.shape[0]):
                    s_idx = float(prop[p, 0].item())
                    e_idx = float(prop[p, 1].item())
                    raw_start = win_start + int(s_idx / T_global_now * win_len)
                    raw_end = win_start + int(e_idx / T_global_now * win_len)
                    raw_end = min(raw_end, win_end)

                    for cl in range(num_classes):
                        score = float(joint_prob[p, cl].item())
                        if score >= conf_thresh:
                            class_outputs[cl].append([raw_start, raw_end, score])

            if device.type == "cuda":
                gpu_mem_list.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

        # Soft-NMS
        final_proposals = []
        for cl in range(num_classes):
            if not class_outputs[cl]:
                continue
            segments = torch.tensor(class_outputs[cl], dtype=torch.float32)
            segments_nms, _ = softnms_v2(
                segments, sigma=nms_sigma, top_k=top_k, score_threshold=conf_thresh
            )
            class_name = test_dataset.id_to_action.get(str(cl), f"class_{cl}")
            for seg in segments_nms:
                final_proposals.append({
                    "label": class_name,
                    "score": float(seg[2]),
                    "segment": [float(seg[0]), float(seg[1])]
                })

        result_dict[file_name] = final_proposals

    # 6) stats + save
    avg_inf_time = float(np.mean(inf_time_list)) if inf_time_list else 0.0
    std_inf_time = float(np.std(inf_time_list)) if inf_time_list else 0.0
    avg_gpu_mem = float(np.mean(gpu_mem_list)) if gpu_mem_list else 0.0
    std_gpu_mem = float(np.std(gpu_mem_list)) if gpu_mem_list else 0.0

    stats_data = {
        "测试文件总数": num_test_files,
        "平均窗口推理时间(ms)": f"{avg_inf_time:.2f} ± {std_inf_time:.2f}",
        "平均GPU峰值内存(MB)": f"{avg_gpu_mem:.2f} ± {std_gpu_mem:.2f}",
        "pretrained_name": pretrained_name,
        "use_airpods": use_airpods,
        "in_channels": in_channels,
        "conf_thresh": conf_thresh,
        "nms_sigma": nms_sigma,
        "top_k": top_k,
        "num_proposals": num_proposals,
        "test_mode": test_mode,
    }
    with open(os.path.join(result_path, "inference_stats_oicr_pa.json"), "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    prediction_data = {
        "version": "OICR-PA-IMU-v1.0",
        "results": result_dict,
        "external_data": {}
    }
    pred_save_path = os.path.join(result_path, f"predictions_{test_mode}.json")
    with open(pred_save_path, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, indent=2, ensure_ascii=False)
    print(f"[OICR-PA] 预测结果已保存至：{pred_save_path}")

    # 7) mAP
    print("\n[OICR-PA] 开始评估动作定位性能...")
    tious = np.linspace(0.3, 0.7, 5)
    anet_evaluator = ANETdetection(
        ground_truth_filename=test_dataset.eval_gt,
        prediction_filename=pred_save_path,
        subset="test",
        tiou_thresholds=tious
    )
    mAPs, avg_mAP, _ = anet_evaluator.evaluate()
    print(f"[ANET] {test_mode} avg_mAP={avg_mAP:.4f}")

    report_content = [
        "=" * 60,
        "OICR-PA-IMU 测试报告",
        "=" * 60,
        f"pretrained_name={pretrained_name}, use_airpods={use_airpods}, in_channels={in_channels}",
        f"test_mode={test_mode}, num_files={num_test_files}",
        "-" * 60,
    ]
    for tiou, mAP in zip(tious, mAPs):
        report_content.append(f"tIoU={tiou:.2f} → mAP={mAP:.4f}")
    report_content.append(f"平均mAP：{avg_mAP:.4f}")
    report_content.append("=" * 60)

    report_save_path = os.path.join(result_path, f"report_{test_mode}_oicr_pa.txt")

    # ===== 指标计算：PRF + confusion + UR/OR/DR/IR/FR/MR =====
    class_names = [test_dataset.id_to_action.get(str(i), f"class_{i}") for i in range(num_classes)]
    conf_thr = float(config.get("testing", {}).get("confusion_tiou", 0.5))

    extra_lines, artifacts = evaluate_extra_metrics_and_plots(
        gt_path=test_dataset.eval_gt,
        pred_path=pred_save_path,
        class_names=class_names,
        tious=tious,
        result_dir=result_path,
        test_mode=test_mode,
        subset_name="test",
        confusion_tiou=conf_thr,
        include_bg=True,
        uodifm_unit="frame",  # 按帧计算
        seconds_per_unit=None,  # frame模式不需要
        cm_value_thresh=0.02,
    )

    report_content.extend(extra_lines)

    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    print("\n" + "\n".join(report_content))
    print(f"\n[OICR-PA] 测试完成！结果目录：{result_path}")



def test_pclHead(config, checkpoint_path,test_mode: str = "test_window"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n 开始测试，测试设备：{device}")
    print(f" 加载模型：{checkpoint_path}")

    use_airpods = bool(config["training"].get("use_airpods", False))
    in_channels = 30 + (6 if use_airpods else 0)
    print(f" use_airpods = {use_airpods}, in_channels = {in_channels}")

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
    model = build_pclHead_model(config, num_classes, device)

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
        "num_proposals_full" if test_mode == "test_full" else "num_proposals_window", 800))
    result_path = config["path"]["result_path"]
    os.makedirs(result_path, exist_ok=True)

    print(f" 全局特征时序长度 T_global：{T_global}")

    proposal_boxes = generate_proposal_boxes(
        T_global=T_global,
        num_proposals=num_proposals
    ).to(device)  # [P,2]

    # 5) 推理
    result_dict = {}
    inf_time_list, gpu_mem_list = [], []

    test_files = list(test_dataset.dataset())
    num_test_files = len(test_files)

    for file_name, data_iterator in tqdm(test_files, desc=" 测试进度", unit="文件", total=num_test_files):
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
        "version": "pclHead-IMU-v1.0",
        "results": result_dict,
        "external_data": {}
    }
    pred_save_path = os.path.join(result_path, f"predictions_{test_mode}.json")
    with open(pred_save_path, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, indent=2, ensure_ascii=False)
    print(f" 预测结果已保存至：{pred_save_path}")

    # 7) mAP
    print("\n 开始评估动作定位性能...")
    tious = np.linspace(0.3, 0.7, 5)
    anet_evaluator = ANETdetection(
        ground_truth_filename=test_dataset.eval_gt,
        prediction_filename=pred_save_path,
        subset='test',
        tiou_thresholds=tious
    )
    mAPs, avg_mAP, _ = anet_evaluator.evaluate()
    print(f"[ANET] {test_mode} avg_mAP={avg_mAP:.4f}")

    report_content = [
        "=" * 60,
        "pclHead-IMU 训练+测试综合报告",
        "=" * 60,
        f"测试配置：use_airpods={use_airpods}, in_channels={in_channels}",
        f"测试文件总数：{num_test_files}",
        "-" * 60,
    ]
    for tiou, mAP in zip(tious, mAPs):
        report_content.append(f"tIoU={tiou:.2f} → mAP={mAP:.4f}")
    report_content.append(f"平均mAP：{avg_mAP:.4f}")
    report_content.append("=" * 60)

    report_save_path = os.path.join(result_path, f"train_{test_mode}_report_pclHead.txt")
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    print("\n" + "\n".join(report_content))
    print(f"\n 测试完成！所有结果已保存至：{result_path}")



@torch.no_grad()
def test_wscnet_imu(config, checkpoint_path):
    """
    使用 WSCNet 的帧级输出 + proposal 池化，导出 ActivityNet 风格的检测结果并评估 mAP。
    流程与 WSDDN 保持一致，只是把帧级分数换成 WSCNet 的输出。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[WSCNet-TEST] device = {device}")
    print(f"[WSCNet-TEST] load checkpoint from: {checkpoint_path}")

    # ----------------------------------------------------------------------
    # 1) 测试数据集
    # ----------------------------------------------------------------------
    test_dataset = WeaklySupervisedXRFV2DatasetTest(
        config=config,
        modality="imu",
        device_keep_list=config["testing"].get("device_keep_list", None),
    )

    # ----------------------------------------------------------------------
    # 2) 标签映射（新ID -> 动作名）
    # ----------------------------------------------------------------------
    _, _, new_to_action = load_label_mapping(config["path"]["mapping_path"])
    num_classes = len(new_to_action)
    print(f"[WSCNet-TEST] num_classes = {num_classes}")

    # ----------------------------------------------------------------------
    # 3) 构建 WSCNet 模型并加载权重（内部会加载预训练 backbone）
    # ----------------------------------------------------------------------
    wscnet = build_wscnet_imu_model(
        config=config,
        num_classes=num_classes
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        wscnet.load_state_dict(ckpt["model_state_dict"])
    else:
        wscnet.load_state_dict(ckpt)
    wscnet.to(device)
    wscnet.eval()
    print("[WSCNet-TEST] WSCNet model loaded.")

    # ----------------------------------------------------------------------
    # 4) dummy 输入跑一次 WSCNet，得到时间长度 T_global
    #    IMUWSCNet.forward 返回: logits_det, logits_cls, class_maps, att_global
    # ----------------------------------------------------------------------
    dummy_input = torch.randn(1, 30, test_dataset.target_len).to(device)
    dummy_out = wscnet(dummy_input)

    if isinstance(dummy_out, (tuple, list)):
        logits_det_d, logits_cls_d, class_maps_d, att_d = dummy_out[:4]
    elif isinstance(dummy_out, dict):
        logits_det_d = dummy_out["logits_det"]
        logits_cls_d = dummy_out["logits_cls"]
        class_maps_d = dummy_out["class_maps"]
        att_d = dummy_out["att_global"]
    else:
        raise RuntimeError("Unsupported WSCNet dummy output format.")

    # class_maps_d: [B, C, T']
    T_global = class_maps_d.shape[-1]
    print(f"[WSCNet-TEST] T_global (feature time length) = {T_global}")

    # ----------------------------------------------------------------------
    # 5) 在 WSCNet 时序长度上生成 proposals
    # ----------------------------------------------------------------------
    num_proposals = config["testing"]["num_proposals"]
    proposal_boxes = generate_proposal_boxes(
        T_global=T_global,
        num_proposals=num_proposals
    ).to(device)  # [P, 2]

    # ----------------------------------------------------------------------
    # 6) 测试超参 & 保存路径
    # ----------------------------------------------------------------------
    conf_thresh = config["testing"]["conf_thresh"]
    nms_sigma = config["testing"]["nms_sigma"]
    top_k = config["testing"]["top_k"]

    result_path = config["path"]["result_path"]
    os.makedirs(result_path, exist_ok=True)
    print(f"[WSCNet-TEST] results will be saved to: {result_path}")

    # 记录推理时间 / 显存
    inf_time_list: List[float] = []
    gpu_mem_list: List[float] = []

    # ----------------------------------------------------------------------
    # 7) 遍历所有测试视频
    # ----------------------------------------------------------------------
    all_results: Dict[str, List[Dict]] = {}
    test_files = list(test_dataset.dataset())

    for file_name, data_iterator in tqdm(test_files, desc="[WSCNet-TEST] videos", unit="video"):
        # 为当前视频收集所有类别的 proposal
        class_outputs = [[] for _ in range(num_classes)]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # data_iterator: yield (clip_dict, [win_start, win_end])
        for clip_dict, (win_start, win_end) in data_iterator:
            imu_clip = clip_dict["imu"].to(device).unsqueeze(0)  # [1, 30, T_in]
            B = imu_clip.shape[0]
            assert B == 1

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_t = time.time()

            # (1) WSCNet 前向
            out = wscnet(imu_clip)
            if isinstance(out, (tuple, list)):
                logits_det, logits_cls, class_maps, att_global = out[:4]
            elif isinstance(out, dict):
                logits_det = out["logits_det"]
                logits_cls = out["logits_cls"]
                class_maps = out["class_maps"]
                att_global = out["att_global"]
            else:
                raise RuntimeError("Unsupported WSCNet test output format.")

            # (2) 融合得到帧级分数
            # video_scores: [B, C], temporal_scores: [B, C, T_global]
            video_scores, temporal_scores = fuse_wscnet_outputs(
                logits_det=logits_det,
                logits_cls=logits_cls,
                class_maps=class_maps,
                att_global=att_global,
            )
            frame_scores = temporal_scores[0]  # [C, T_global]

            if device.type == "cuda":
                torch.cuda.synchronize()
            inf_time = (time.time() - start_t) * 1000.0
            inf_time_list.append(inf_time)

            # (3) proposal 内池化：0.7 * max + 0.3 * mean
            clip_len = test_dataset.clip_length  # 原始窗口长度（例如 2048）
            for p in range(num_proposals):
                s_idx, e_idx = proposal_boxes[p]
                s_idx = int(s_idx.item())
                e_idx = int(e_idx.item())
                if e_idx <= s_idx:
                    continue

                seg_scores = frame_scores[:, s_idx:e_idx]  # [C, L_p]
                if seg_scores.numel() == 0:
                    continue

                score_max = seg_scores.max(dim=1).values   # [C]
                score_mean = seg_scores.mean(dim=1)        # [C]
                proposal_score = 0.7 * score_max + 0.3 * score_mean  # [C]

                # (4) 映射到原始时间轴（单位：原始采样点）
                feat_len = e_idx - s_idx
                raw_len = int(feat_len / T_global * clip_len)

                raw_start = win_start + int(s_idx / T_global * clip_len)
                raw_end = raw_start + raw_len
                raw_end = min(raw_end, win_end)

                # (5) 为每个类别收集 proposal
                for c in range(num_classes):
                    sc = float(proposal_score[c].item())
                    if sc <= conf_thresh:
                        continue
                    class_outputs[c].append([raw_start, raw_end, sc])

            if device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                gpu_mem_list.append(peak_mem)

        # ------------------------------------------------------------------
        # 8) 当前视频的所有类别做 Soft-NMS
        # ------------------------------------------------------------------
        video_proposals: List[Dict] = []
        for c in range(num_classes):
            if not class_outputs[c]:
                continue
            segments = torch.tensor(class_outputs[c], dtype=torch.float32)  # [N, 3]
            segments_nms, _ = softnms_v2(
                segments,
                sigma=nms_sigma,
                top_k=top_k,
                score_threshold=conf_thresh,
            )
            class_name = new_to_action[c]
            for seg in segments_nms:
                s, e, sc = seg.tolist()
                video_proposals.append(
                    {
                        "label": class_name,
                        "score": float(sc),
                        "segment": [float(s), float(e)],
                    }
                )

        all_results[file_name] = video_proposals

    # ----------------------------------------------------------------------
    # 9) 保存推理性能统计
    # ----------------------------------------------------------------------
    avg_inf_time = float(np.mean(inf_time_list)) if inf_time_list else 0.0
    std_inf_time = float(np.std(inf_time_list)) if inf_time_list else 0.0
    avg_gpu_mem = float(np.mean(gpu_mem_list)) if gpu_mem_list else 0.0
    std_gpu_mem = float(np.std(gpu_mem_list)) if gpu_mem_list else 0.0

    stats = {
        "num_test_videos": len(test_files),
        "avg_window_inference_time_ms": f"{avg_inf_time:.2f} ± {std_inf_time:.2f}",
        "avg_gpu_peak_memory_MB": f"{avg_gpu_mem:.2f} ± {std_gpu_mem:.2f}",
        "conf_thresh": conf_thresh,
        "nms_sigma": nms_sigma,
        "num_proposals": num_proposals,
    }
    with open(os.path.join(result_path, "wscnet_inference_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------------------
    # 10) 保存 ActivityNet 风格预测 JSON
    # ----------------------------------------------------------------------
    pred_json = {
        "version": "WSCNet-IMU-v1.0",
        "results": all_results,
        "external_data": {},
    }
    pred_save_path = os.path.join(result_path, "wscnet_proposal_predictions.json")
    with open(pred_save_path, "w", encoding="utf-8") as f:
        json.dump(pred_json, f, indent=2, ensure_ascii=False)
    print(f"[WSCNet-TEST] proposal-based predictions saved to: {pred_save_path}")

    # ----------------------------------------------------------------------
    # 11) ActivityNet 风格评估 (tIoU-mAP)
    # ----------------------------------------------------------------------
    print("\n[WSCNet-TEST] start ActivityNet-style evaluation ...")
    tious = np.linspace(0.1, 0.9, 9)
    anet_eval = ANETdetection(
        ground_truth_filename=test_dataset.eval_gt,
        prediction_filename=pred_save_path,
        subset="test",
        tiou_thresholds=tious,
    )
    mAPs, avg_mAP, _ = anet_eval.evaluate()

    report_lines = [
        "=" * 60,
        "WSCNet-IMU Test Report (proposal-based)",
        "=" * 60,
        f"Prediction file: {pred_save_path}",
        "",
        "Detection mAP (per tIoU):",
        "-" * 60,
    ]
    for t, ap in zip(tious, mAPs):
        report_lines.append(f"tIoU={t:.2f} -> mAP={ap:.4f}")
    report_lines.extend(
        [
            "-" * 60,
            f"Average mAP: {avg_mAP:.4f}",
            "=" * 60,
        ]
    )

    report_path = os.path.join(result_path, "wscnet_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"[WSCNet-TEST] evaluation report saved to: {report_path}")

    return pred_save_path, report_path
