# -*- coding: utf-8 -*-
import torch
import os
import json
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保 dataset_xrfv2.py 在同级目录
from dataset.dataset_xrfv2 import WeaklySupervisedXRFV2DatasetTrain, WeaklySupervisedXRFV2DatasetTest
from models.CDur_model import CDur
from tool import ANETdetection

# 基础环境设置
torch.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# ============================================================
# 1) 损失函数：解决 Loss 0.0000 的问题
# ============================================================
class RobustBCELoss(torch.nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, clip_prob, labels):
        # clip_prob 已经被模型内部 clamp 到 [1e-7, 1]
        n_classes = clip_prob.shape[-1]
        with torch.no_grad():
            # 标签平滑
            target = labels * (1 - self.label_smoothing) + (1 - labels) * self.label_smoothing / (n_classes - 1)

        # 使用 binary_cross_entropy，显式增加 reduction='mean'
        # 如果依然显示 0.0000，说明预测值和 target 极其接近
        loss = F.binary_cross_entropy(clip_prob, target)
        return loss


def frame_probs_to_segments(probs, fps, threshold=0.1, min_duration=0.2):
    """
    针对弱监督任务，阈值 threshold 通常设得较低 (如 0.1-0.2)
    """
    T, C = probs.shape
    segments = [[] for _ in range(C)]
    for c in range(C):
        binary = probs[:, c] > threshold
        diff = np.diff(np.concatenate(([0], binary.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            t_start = s / fps
            t_end = e / fps
            if (t_end - t_start) >= min_duration:
                score = np.mean(probs[s:e, c])
                segments[c].append([t_start, t_end, float(score)])
    return segments


# ============================================================
# 2) 训练函数：适配 CDur 的 Forward 逻辑
# ============================================================
def train_cdur_xrfv2(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_airpods = config["training"]["use_airpods"]
    in_channels = 36 if use_airpods else 30
    num_classes = config["training"]["num_classes"]

    # 初始化模型：CDur 内部会处理 [B, T, C] -> [B, C, T]
    model = CDur(inputdim=in_channels, outputdim=num_classes,
                 temppool=config["model"]["temppool"]).to(device)

    train_ds = WeaklySupervisedXRFV2DatasetTrain(
        dataset_dir=config["path"]["train_dataset_path"],
        mapping_path=config["path"]["mapping_path"],
        split="train",
        use_airpods=use_airpods
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )

    # 学习率建议：CDur 这种 RNN 结构，lr=1e-4 是比较稳妥的
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=1e-5)
    criterion = RobustBCELoss(label_smoothing=0.05).to(device)

    os.makedirs(config["path"]["checkpoint_path"], exist_ok=True)
    best_loss = float('inf')

    print(f">>> XRFV2 Training Started | Input: {in_channels} | Classes: {num_classes}")

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        for batch_idx, (data, _, labels) in enumerate(pbar):
            # 重要：Dataset 返回 [B, 36, 2048]，即 [B, Dim, Time]
            # 但是 CDur 模型内部 forward 第一行是 x = x.transpose(1, 2)
            # 所以我们必须在这里把 data 换成 [B, Time, Dim] 传进去
            data = data.transpose(1, 2).to(device)  # 变为 [B, 2048, 36]
            labels = labels.to(device)

            optimizer.zero_grad()
            clip_prob, _ = model(data)  # 模型内部会转回 [B, 36, 2048] 进行卷积

            loss = criterion(clip_prob, labels)

            if torch.isnan(loss):
                print("Warning: NaN Loss detected!")
                continue

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # 实时更新进度条显示 Loss 详情，观察是否真的为 0
            pbar.set_postfix({"loss": f"{loss.item():.7f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.7f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(config["path"]["checkpoint_path"], "best_model.pth"))

    return os.path.join(config["path"]["checkpoint_path"], "best_model.pth")


# ============================================================
# 3) 测试函数
# ============================================================
# ============================================================
# 修改后的测试分发函数
# ============================================================

def soft_nms_functional(dets, sigma=0.5, thresh=0.001):
    """
    dets: [[start, end, score], ...]
    """
    if len(dets) == 0: return []

    tstart = dets[:, 0]
    tend = dets[:, 1]
    tscore = dets[:, 2]

    order = tscore.argsort()[::-1]
    res_indices = []

    while order.size > 0:
        i = order[0]
        res_indices.append(i)
        if order.size == 1: break

        # 计算交集
        xx1 = np.maximum(tstart[i], tstart[order[1:]])
        xx2 = np.minimum(tend[i], tend[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)

        # 计算并集 (IoU)
        areas = tend - tstart
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union

        # Soft-NMS 衰减分数
        weight = np.exp(-(iou * iou) / sigma)
        tscore[order[1:]] *= weight

        # 过滤掉分数过低的
        mask = tscore[order[1:]] > thresh
        order = order[1:][mask]
        # 重新排序
        if order.size > 0:
            new_order = tscore[order].argsort()[::-1]
            order = order[new_order]

    return res_indices


@torch.no_grad()
def test_cdur_xrfv2(config, checkpoint_path, test_mode="test_window"):
    """
    支持 test_window 和优化后的 test_full（聚合滑窗预测 + NMS）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_airpods = config["training"].get("use_airpods", True)
    in_channels = 36 if use_airpods else 30
    num_classes = config["training"]["num_classes"]
    fps = 50

    # 1. 加载模型
    model = CDur(inputdim=in_channels, outputdim=num_classes,
                 temppool=config["model"].get("temppool", "linear")).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. 初始化数据集
    test_ds = WeaklySupervisedXRFV2DatasetTest(config=config, use_airpods=use_airpods)
    id2label = test_ds.id_to_action

    # 调优参数：Full 模式下建议降低阈值以提高召回，交给 NMS 去重
    conf_thresh = config["testing"].get("conf_thresh", 0.01 if test_mode == "test_full" else 0.05)
    nms_sigma = config["testing"].get("nms_sigma", 0.3)

    results_cache = {}
    print(f"\n>>> 开始测试模式: {test_mode}")

    # 3. 推理循环
    for file_path_raw, data_iter in tqdm(test_ds.dataset(), desc=f"Testing {test_mode}"):
        video_id = os.path.basename(file_path_raw)

        # 建立临时列表存储当前视频的所有原始候选框（按类别）
        raw_predictions = [[] for _ in range(num_classes)]

        for clip_dict, seg_range in data_iter:
            x = clip_dict['imu'].T.unsqueeze(0).to(device)
            _, frame_prob = model(x, upsample=True)
            frame_prob = frame_prob.squeeze(0).cpu().numpy()

            # 获取该窗口内的动作片段
            segments = frame_probs_to_segments(frame_prob, fps, threshold=conf_thresh)
            offset_frames = seg_range[0]

            for cls_idx, segs in enumerate(segments):
                for (s_sec, e_sec, score) in segs:
                    # 将窗口内的时间映射到全局帧数
                    start_frame = s_sec * fps + offset_frames
                    end_frame = e_sec * fps + offset_frames
                    raw_predictions[cls_idx].append([start_frame, end_frame, score])

        # 4. 后处理聚合
        final_video_preds = []

        if test_mode == "test_full":
            # 对每个类别执行 Soft-NMS，合并不同窗口产生的重叠检测
            for cls_idx in range(num_classes):
                cls_segs = np.array(raw_predictions[cls_idx])
                if len(cls_segs) == 0: continue

                # 执行您之前添加的 soft_nms_functional
                keep_indices = soft_nms_functional(cls_segs, sigma=nms_sigma, thresh=conf_thresh)

                label_name = id2label.get(str(cls_idx), id2label.get(cls_idx, f"class_{cls_idx}"))
                for i in keep_indices:
                    final_video_preds.append({
                        "label": label_name,
                        "score": float(cls_segs[i, 2]),
                        "segment": [float(cls_segs[i, 0]), float(cls_segs[i, 1])]
                    })
        else:
            # 传统的 test_window 逻辑：直接保留所有窗口的检测结果（不做全局 NMS）
            for cls_idx in range(num_classes):
                label_name = id2label.get(str(cls_idx), id2label.get(cls_idx, f"class_{cls_idx}"))
                for seg in raw_predictions[cls_idx]:
                    final_video_preds.append({
                        "label": label_name,
                        "score": float(seg[2]),
                        "segment": [float(seg[0]), float(seg[1])]
                    })

        results_cache[video_id] = final_video_preds

    # 5. 构建并保存符合 ActivityNet 规范的预测文件
    output_data = {
        "version": "VERSION 1.3",
        "results": results_cache,
        "external_data": {}
    }

    res_name = f"prediction_{test_mode}.json"
    pred_path = os.path.join(config["path"]["result_path"], res_name)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)

    with open(pred_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_preds = sum(len(v) for v in results_cache.values())
    print(f"[{test_mode}] 总共生成检测框数量: {total_preds}")

    if total_preds == 0:
        print(f"警告: {test_mode} 模式下没有检测到任何动作。")
        return 0.0

    # 6. 计算 mAP
    tious = [0.1, 0.2, 0.3, 0.4, 0.5]
    evaluator = ANETdetection(
        ground_truth_filename=test_ds.eval_gt,
        prediction_filename=pred_path,
        subset="test",
        tiou_thresholds=tious,
        verbose=False,
        check_status=False
    )

    mAPs, avg_mAP, _ = evaluator.evaluate()
    print(f"[{test_mode}] Average mAP: {avg_mAP:.4f}")
    return avg_mAP


# ============================================================
# 更新后的 Main 结构
# ============================================================
if __name__ == "__main__":
    # 使用你刚才提供的 base_config 风格进行配置
    config = {
        "path": {
            "train_dataset_path": "/home/lipei/XRFV2/",
            "test_dataset_path": "/home/lipei/XRFV2/",
            "dataset_root_path": "/home/lipei/WWADL/",
            "mapping_path": "/home/lipei/project/WSDDN/label_mapping.json",
            "checkpoint_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/checkpoints/xrfv2_cdur",
            "result_path": "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_cdur"
        },
        "model": {"temppool": "linear"},
        "training": {
            "use_airpods": True,
            "num_classes": 30,
            "batch_size": 16,
            "num_epochs": 80,
            "lr": 1e-3,
            "num_workers": 4
        },
        "testing": {
            "conf_thresh": 0.05,
        }
    }

    # 1. 训练
    best_ckpt = train_cdur_xrfv2(config)

    # 2. 双模式测试 (参考 main() 中的顺序)
    test_cdur_xrfv2(config, best_ckpt, test_mode="test_full")
    test_cdur_xrfv2(config, best_ckpt, test_mode="test_window")
