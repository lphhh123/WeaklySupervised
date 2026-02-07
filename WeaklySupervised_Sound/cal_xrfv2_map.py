import numpy as np
from tool import ANETdetection  # 假设你上面的代码保存在 tool.py 中


def calculate_metrics(gt_path, pred_path):
    # 1. 初始化评估器
    # tiou_thresholds: 评估的 IoU 阈值，通常 DCASE 或 ActivityNet 使用 0.5 到 0.95
    t_iou = np.linspace(0.3, 0.7, 5)

    # 注意：subset 必须与 GT 文件中的 "subset": "test" 一致
    evaluator = ANETdetection(
        ground_truth_filename=gt_path,
        prediction_filename=pred_path,
        tiou_thresholds=t_iou,
        subset='test',
        verbose=True,
        check_status=False
    )

    # 2. 执行评估
    # mAP: 每个阈值下的平均精度 (1D array)
    # average_mAP: 所有阈值下的平均值 (scalar)
    # ap: 每个类别在每个阈值下的精度 (2D array [num_thresholds, num_classes])
    mAP, average_mAP, ap = evaluator.evaluate()

    print("\n" + "=" * 30)
    print(f"Final Average mAP: {average_mAP:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    # 请确保路径正确
    GT_FILE = "/home/lipei/XRFV2/imu_annotations.json"  # 假设你的GT文件名
    PRED_FILE = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_cdur/prediction_test_full.json"

    calculate_metrics(GT_FILE, PRED_FILE)