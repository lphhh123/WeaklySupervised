import numpy as np
from tool import ANETdetection


def calculate_metrics(gt_path, pred_path):
    t_iou = np.linspace(0.3, 0.7, 5)

    evaluator = ANETdetection(
        ground_truth_filename=gt_path,
        prediction_filename=pred_path,
        tiou_thresholds=t_iou,
        subset='test',
        verbose=True,
        check_status=False
    )

    mAP, average_mAP, ap = evaluator.evaluate()

    print("\n" + "=" * 30)
    print(f"Final Average mAP: {average_mAP:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    GT_FILE = "/home/lipei/XRFV2/imu_annotations.json"
    PRED_FILE = "/home/yinjiaxi/wstal/WeaklySupervised-master/result/xrfv2_cdur/prediction_test_full.json"

    calculate_metrics(GT_FILE, PRED_FILE)