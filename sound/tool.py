import os
from typing import Dict, Type, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
from tqdm import tqdm

from joblib import Parallel, delayed
import pandas as pd
import random

def load_label_mapping(mapping_path):
    """
    读取label_mapping.json，返回：
    - id_to_action: {旧ID: 动作名}
    - old_to_new: {旧ID: 新ID}
    - new_to_action: {新ID: 动作名}（新增：方便测试时映射）
    """
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    # 旧ID→动作名（键转为int）
    id_to_action = {int(k): v for k, v in mapping["id_to_action"].items()}
    # 旧ID→新ID（键转为int）
    old_to_new = {int(k): v for k, v in mapping["old_to_new_mapping"].items()}
    # 新ID→动作名（去重合并）
    new_to_action = {}
    for old_id, act_name in id_to_action.items():
        new_id = old_to_new[old_id]
        if new_id not in new_to_action:
            new_to_action[new_id] = act_name  # 保留第一个映射的动作名（同组动作名相似）

    return id_to_action, old_to_new, new_to_action


class ANETdetection(object):
    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 subset='validation', verbose=False,
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.

        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()

        # Import ground truth and predictions.
        self.ground_truth, self.activity_index, self.video_lst = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        action_druation = []
        for videoid, v in data['database'].items():
            # print(v)
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])
                duration = int(float(ann['segment'][1]) - float(ann['segment'][0]))

                if duration < 8 * 50:
                    action_druation.append(0)
                elif 8 * 50 <= duration < 12 * 50:
                    action_druation.append(1)
                else:
                    action_druation.append(2)

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst,
                                     'duration_type': action_druation})
        if self.verbose:
            print(activity_index)
        return ground_truth, activity_index, video_lst

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        action_druation = []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            if videoid not in self.video_lst:
                continue
            for result in v:
                if result['label'] not in self.activity_index:
                    continue
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
                duration = int(float(result['segment'][1]) - float(result['segment'][0]))

                if duration < 8 * 50:
                    action_druation.append(0)
                elif 8 * 50 <= duration < 12 * 50:
                    action_druation.append(1)
                else:
                    action_druation.append(2)

        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst,
                                   'duration_type': action_druation})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            if self.verbose:
                print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_index))(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                label_name=label_name,
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()
        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        clsmap = self.ap.mean(axis=0)

        # Loop through each action category and print the mAP for each tIoU threshold
        for idx, category in enumerate(self.activity_index.keys()):
            # Extract mAP for each tIoU threshold for the current category
            category_map_values = " & ".join([f"{self.ap[i, idx]:.4f}" for i in range(len(self.tiou_thresholds))])

            # Calculate the mean mAP for the category across all thresholds
            mean_map = clsmap[idx]

            # Print the LaTeX formatted row for the current action category
            print(f"{category} & {category_map_values} & {mean_map:.4f} \\\\")

        if self.verbose:
            print('[RESULTS] Performance on ActivityNet detection task.')
            print('Average-mAP: {}'.format(self.average_mAP))

        return self.mAP, self.average_mAP, self.ap


import json
import urllib.request


API = 'http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/challenge17/api.py'

def get_blocked_videos(api=API):
    api_url = '{}?action=get_blocked'.format(api)
    req = urllib.request.Request(api_url)
    response = urllib.request.urlopen(req)
    return json.loads(response.read().decode('utf-8'))


def compute_average_precision_detection(ground_truth, prediction, label_name, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)

        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                if len(ground_truth) == 1:
                    print(f'tiou_arr: {tiou_arr[jdx]} {label_name} {tiou_thr} {idx}')
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)
    # if len(ground_truth) == 1:
    #     print(f'precision_cumsum: {precision_cumsum}')

    # if len(ground_truth) == 1:
    #     # Directly calculate precision for each threshold and store in ap[]
    #     for tidx in range(len(tiou_thresholds)):
    #         precision_at_threshold = precision_cumsum[tidx, 0]  # For each threshold (0.75, 0.8, 0.85, ..., 0.95)
    #         ap[tidx] = precision_at_threshold  # Assign precision to ap for each threshold

    #     print(f"Precision values for Lying Still at each threshold: {ap}")
    # else:
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU



def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def softnms_v2(segments, sigma=0.5, top_k=1000, score_threshold=0.001):
    segments = segments.cpu()
    tstart = segments[:, 0]
    tend = segments[:, 1]
    tscore = segments[:, 2]
    done_mask = tscore < -1  # set all to False
    undone_mask = tscore >= score_threshold
    while undone_mask.sum() > 1 and done_mask.sum() < top_k:
        idx = tscore[undone_mask].argmax()
        idx = undone_mask.nonzero()[idx].item()

        undone_mask[idx] = False
        done_mask[idx] = True

        top_start = tstart[idx]
        top_end = tend[idx]
        _tstart = tstart[undone_mask]
        _tend = tend[undone_mask]
        tt1 = _tstart.clamp(min=top_start)
        tt2 = _tend.clamp(max=top_end)
        intersection = torch.clamp(tt2 - tt1, min=0)
        duration = _tend - _tstart
        tmp_width = torch.clamp(top_end - top_start, min=1e-5)
        iou = intersection / (tmp_width + duration - intersection)
        scales = torch.exp(-iou ** 2 / sigma)
        tscore[undone_mask] *= scales
        undone_mask[tscore < score_threshold] = False
    count = done_mask.sum()
    segments = torch.stack([tstart[done_mask], tend[done_mask], tscore[done_mask]], -1)
    return segments, count


