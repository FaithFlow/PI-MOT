# ------------------------------------------------------------------------
# Copyright (c) 2026 PI-MOT. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from CO-MOT, MOTR, DAB-DETR, Deformable DETR, and DETR.
# Original copyright notices from the upstream projects are retained where
# applicable in the corresponding source files.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch.distributed as dist
import torch
import util.misc as utils
import numpy as np
from datasets.data_prefetcher import data_dict_to_cuda

import cv2
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torchvision.transforms.functional as F


DEFAULT_TRACK_EVAL_SCRIPT_DIR = "/home/member/data2/jsk/code/TrackEval/scripts"
DEFAULT_TRACK_EVAL_CONFIGS = {
    'e2e_mot20': {
        'gt_folder': "/home/member/data2/jsk/datasets/MOT20/train/",
        'seqmap_file': "/home/member/data2/jsk/datasets/MOT20/train_seqmap.txt",
        'benchmark': None,
    },
    'e2e_sports': {
        'gt_folder': "/home/member/data2/jsk/datasets/SportsMOT/val/",
        'seqmap_file': "/home/member/data2/jsk/datasets/SportsMOT/val_seqmap.txt",
        'benchmark': None,
    },
    'e2e_dance': {
        'gt_folder': "/home/member/data2/jsk/datasets/DanceTrack/val",
        'seqmap_file': "/home/member/data2/jsk/datasets/DanceTrack/val_seqmap.txt",
        'benchmark': None,
    },
}


def get_trackeval_config(args):
    config = DEFAULT_TRACK_EVAL_CONFIGS.get(args.dataset_file, DEFAULT_TRACK_EVAL_CONFIGS['default']).copy()
    overrides = {
        'gt_folder': getattr(args, 'trackeval_gt_folder', None),
        'seqmap_file': getattr(args, 'trackeval_seqmap_file', None),
        'benchmark': getattr(args, 'trackeval_benchmark', None),
    }
    for key, value in overrides.items():
        if value:
            config[key] = value
    return config


def run_trackeval(args, tracker_dir):
    script_dir = getattr(args, 'trackeval_script_dir', None) or DEFAULT_TRACK_EVAL_SCRIPT_DIR
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    import run_mot_challenge

    config = get_trackeval_config(args)
    eval_kwargs = {
        'SPLIT_TO_EVAL': "val",
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'GT_FOLDER': config['gt_folder'],
        'SEQMAP_FILE': config['seqmap_file'],
        'SKIP_SPLIT_FOL': True,
        'TRACKERS_TO_EVAL': [''],
        'TRACKER_SUB_FOLDER': '',
        'USE_PARALLEL': True,
        'NUM_PARALLEL_CORES': 8,
        'PLOT_CURVES': False,
    }
    if config['benchmark'] is not None:
        eval_kwargs['BENCHMARK'] = config['benchmark']

    eval_kwargs['TRACKERS_FOLDER'] = tracker_dir
    res_eval = run_mot_challenge.main(**eval_kwargs)

    return float(np.mean(res_eval[0]['MotChallenge2DBox']['']['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']))

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1000

    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        data_dict['epoch'] = epoch
        outputs = model(data_dict)

        loss_dict = criterion(outputs, data_dict)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, skipping this iteration".format(loss_value))
            print(loss_dict_reduced)
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        if torch.isnan(grad_total_norm).any():
            print(data_dict['gt_instances'])
            optimizer.zero_grad()
            continue

        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class ListImgDataset(Dataset):
    def __init__(self, img_list) -> None:
        super().__init__()
        self.img_list = img_list
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(f_path)
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)

        return cur_img, f_path

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale / 32) * 32
        target_w = int(self.seq_w * scale / 32) * 32
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, f_path = self.load_img_from_file(self.img_list[index])
        img, ori_img = self.init_img(img)
        return img, ori_img, f_path


def filter_dt_by_score(dt_instances, prob_threshold):
    keep = dt_instances.scores > prob_threshold
    keep &= dt_instances.obj_idxes >= 0
    return dt_instances[keep]

def filter_dt_by_dual_score(dt_instances, prob_threshold):
    keep = (dt_instances.scores > prob_threshold) * (dt_instances.invididual > prob_threshold)
    keep &= dt_instances.obj_idxes >= 0
    return dt_instances[keep]

def filter_dt_by_area(dt_instances, area_threshold):
    wh = dt_instances.boxes[..., 2:4] - dt_instances.boxes[..., 0:2]
    areas = wh[..., 0] * wh[..., 1]
    keep = areas > area_threshold
    return dt_instances[keep]

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, output_dir, args=None):
    model.eval()
    criterion.eval()
    model_ref = model.module if hasattr(model, "module") else model
    tracker_dir = os.path.join(output_dir, 'tracker')

    prob_threshold = args.score_threshold
    area_threshold = 100

    for data_dict in data_loader:
        print(data_dict)
        
        seq_num = os.path.basename(data_dict['video_name'][0])
        img_list = os.listdir(os.path.join(data_dict['video_name'][0], 'img1'))
        img_list = [os.path.join(data_dict['video_name'][0], 'img1', i) for i in img_list if 'jpg' in i]
        
        img_list = sorted(img_list)
        
        track_instances = None
        loader = DataLoader(ListImgDataset(img_list), 1, num_workers=2)
        lines = defaultdict(list)
        total_dts = defaultdict(int)
        total_occlusion_dts = defaultdict(int)
        record_name = ''

        for i, data in enumerate(loader):
            cur_img, ori_img, f_path = [d[0] for d in data]
            cur_img = cur_img.to(device)
            seq_name = f_path.split('/')[-3]
            if record_name != seq_name:
                record_name = seq_name
                track_instances = None

            if track_instances is not None:
                track_instances.remove('boxes')
            seq_h, seq_w, _ = ori_img.shape

            res = model_ref.inference_single_image(cur_img, (seq_h, seq_w), track_instances)
            track_instances = res['track_instances']

            dt_instances_all = deepcopy(track_instances).get_bn(0)
            dt_instances_all = filter_dt_by_score(dt_instances_all, prob_threshold)
            dt_instances_all = filter_dt_by_area(dt_instances_all, area_threshold)
            active_indx = []
            full_indx = torch.arange(len(dt_instances_all), device=dt_instances_all.scores.device)
            for id in torch.unique(dt_instances_all.obj_idxes):
                indx = torch.where(dt_instances_all.obj_idxes == id)[0]
                active_indx.append(full_indx[indx][dt_instances_all.scores[indx].argmax()])
            if len(active_indx):
                active_indx = torch.stack(active_indx)
                dt_instances_all = dt_instances_all[active_indx]
            
            dt_instances = dt_instances_all
            total_dts[0] += len(dt_instances)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()
            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                if args.dataset_file == 'e2e_mot':
                    frame_ith = int(os.path.splitext(os.path.basename(f_path))[0])
                    lines[0].append(save_format.format(frame=frame_ith, id=track_id, x1=x1, y1=y1, w=w, h=h))
                else:
                    lines[0].append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))
                    
        os.makedirs(tracker_dir, exist_ok=True)
        with open(os.path.join(tracker_dir, f'{seq_num}.txt'), 'w') as f:
            f.writelines(lines[0])
        print("{}: totally {} dts {} occlusion dts".format(seq_num, total_dts[0], total_occlusion_dts[0]))
    #'''
    if dist.is_initialized():
        dist.barrier()
    return run_trackeval(args, tracker_dir)
