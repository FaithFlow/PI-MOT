# ------------------------------------------------------------------------
# Copyright (c) 2026 PI-MOT. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from CO-MOT, MOTR, DAB-DETR, Deformable DETR, and DETR.
# Original copyright notices from the upstream projects are retained where
# applicable in the corresponding source files.
# ------------------------------------------------------------------------


import argparse
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.load_asso import load_model

import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch_mot, evaluate
from models import build_model
import torch.backends.cudnn as cudnn

cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    print(f"Initializing worker {worker_id} with seed {worker_seed}")
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def freeze_named_parameters(model, frozen_keywords):
    for name, parameter in model.named_parameters():
        if any(keyword in name for keyword in frozen_keywords):
            parameter.requires_grad = False


def get_args_parser():
    parser = argparse.ArgumentParser('PI-MOT Configer', add_help=False)

    # ===== Active: directly used by main_modulization.py =====
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--output_dir', default='tmp/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
    parser.add_argument('--not_valid', action='store_false', default=True)

    # ===== Active: model dispatch and target MOTR build() =====
    parser.add_argument('--meta_arch', default='motr_batch_dab_yolox', type=str)
    parser.add_argument('--det_nms', default=0.7, type=float)
    parser.add_argument('--qualified_threshold', default=0.2, type=float)
    parser.add_argument('--ada_hist', action='store_true')
    parser.add_argument('--sigma', default=0.2, type=float)
    parser.add_argument('--query_interaction_layer', default='QIM', type=str, help="QIM")
    parser.add_argument('--random_drop', type=float, default=0)
    parser.add_argument('--fp_ratio', type=float, default=0)
    parser.add_argument('--score_threshold', default=0.3, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # ===== Active: dataset builders and engine/evaluate =====
    parser.add_argument('--dataset_file', default='e2e_dance')
    parser.add_argument('--mot_path', default='/data/Dataset/mot', type=str)
    parser.add_argument('--training_set', default='sub', type=str)
    parser.add_argument('--sample_mode', type=str, default='fixed_interval')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--sampler_steps', type=int, nargs='*')
    parser.add_argument('--sampler_lengths', type=int, nargs='*')
    parser.add_argument('--append_crowd', default=False, action='store_true')
    parser.add_argument('--trackeval_script_dir', default=None, type=str)
    parser.add_argument('--trackeval_gt_folder', default=None, type=str)
    parser.add_argument('--trackeval_seqmap_file', default=None, type=str)
    parser.add_argument('--trackeval_benchmark', default=None, type=str)

    #'''
    return parser


def main(args):
    N = 16
    torch.set_num_threads(N)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)
    
    set_random_seed(args.seed + utils.get_rank())
    
    model, criterion = build_model(args)
    model.to(device)
    model_without_ddp = model

    freeze_named_parameters(model_without_ddp, frozen_keywords=('backbone', 'detector'))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
        sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    collate_fn = utils.mot_collate_fn
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True, worker_init_fn=seed_worker)
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
                                 pin_memory=True, worker_init_fn=seed_worker)

    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad and p.is_leaf]
    print('trainable params:', len(trainable_params))

    if args.sgd:
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    if args.pretrained is not None:
        load_model(model_without_ddp, args.pretrained)
    
    output_dir = Path(args.output_dir)

    if 0:
        print('start evaluation')
        t0 = time.time()
        hota = evaluate(model, criterion, data_loader_val, device, args.output_dir, args=args)
        print(hota)
        print('finish evaluation, time:', time.time() - t0)
    print("Start training: %d" % args.start_epoch)
    start_time = time.time()

    dataset_train.set_epoch(args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch_mot(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.not_valid:
            save_middle_ckpt = True
            if save_middle_ckpt and (epoch + 1) >= args.lr_drop[1]:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, os.path.join(output_dir, 'checkpoint_%s.pth'%(str(epoch))))
                
            hota = evaluate(model, criterion, data_loader_val, device, args.output_dir, args=args)
            print('HOTA:', hota)

        dataset_train.step_epoch()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PI-MOT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
