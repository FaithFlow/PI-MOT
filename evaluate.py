# Copyright (c) 2026 PI-MOT. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from CO-MOT, MOTR, DAB-DETR, Deformable DETR, and DETR.
# Original copyright notices from the upstream projects are retained where
# applicable in the corresponding source files.
# ------------------------------------------------------------------------

"""
Standalone evaluation entrypoint.

The tracking/evaluation loop lives in engine.py; this file only prepares args,
model, checkpoint, and validation/test dataloader.
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import datasets.samplers as samplers
from datasets import build_dataset
from engine import evaluate
from main import freeze_named_parameters, seed_worker, set_random_seed
from models import build_model
from util.eval_tool import load_model
import util.misc as utils


cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _has_option(parser, option):
    return any(option in action.option_strings for action in parser._actions)


def get_args_parser():
    from main import get_args_parser as get_train_args_parser

    parser = get_train_args_parser()

    # Compatibility args used by eval/model/dataset code but not always needed
    # during the current training entrypoint.
    extra_args = [
        ('--test_mode', {'default': 'val', 'choices': ('val', 'test'),
                         'help': 'dataset split to evaluate'}),
    ]
    for option, kwargs in extra_args:
        if not _has_option(parser, option):
            parser.add_argument(option, **kwargs)

    return parser


def build_eval_dataloader(args):
    image_set = 'test' if args.test_mode == 'test' else 'val'
    dataset_val = build_dataset(image_set=image_set, args=args)

    if args.distributed:
        sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    return DataLoader(
        dataset_val,
        batch_size=1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.mot_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )


def main(args):
    torch.set_num_threads(16)
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

    data_loader_val = build_eval_dataloader(args)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module


    if args.pretrained is not None:
        load_model(model_without_ddp, args.pretrained)

    print('start evaluation')
    start_time = time.time()
    hota = evaluate(model, criterion, data_loader_val, device, args.output_dir, args=args)
    print('HOTA:', hota)
    print('finish evaluation, time:', time.time() - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PI-MOT evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
