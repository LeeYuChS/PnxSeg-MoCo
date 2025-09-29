"""
cd I:/000WTF/unet_pipeline_moco
conda run --live-stream --name classMAI python Train.py experiments/deeplabv3plus_wavelet/train_config_deeplabv3plus_part0.yaml
conda run --live-stream --name classMAI python Train.py experiments/deeplabv3plus_wavelet/train_config_deeplabv3plus_part1.yaml
conda run --live-stream --name classMAI python Train.py experiments/deeplabv3plus_wavelet/train_config_deeplabv3plus_part2.yaml
"""


import argparse
import logging
import time
import datetime

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import cv2
import torch

import importlib
import functools
from tqdm import tqdm
import os
from pathlib import Path

from Pneumadataset import PneumothoraxDataset, PneumoSampler
from Learning import Learning
from utils.helpers import load_yaml, init_seed, init_logger

class SimpleResize:
    """Simple resize transform to replace albumentations"""
    def __init__(self, height=512, width=512):
        self.height = height
        self.width = width
    
    def __call__(self, **kwargs):
        image = kwargs.get('image')
        mask = kwargs.get('mask')
        
        result = {}
        if image is not None:
            result['image'] = cv2.resize(image, (self.width, self.height))
        if mask is not None:
            result['mask'] = cv2.resize(mask, (self.width, self.height))
        
        return result

# from Evaluation import apply_deep_thresholds, search_deep_thresholds, dice_round_fn, search_thresholds


def argparser():
    parser = argparse.ArgumentParser(description='Pneumatorax pipeline')
    parser.add_argument('train_cfg', type=str, help='train config path')
    return parser.parse_args()

    
def train_fold(
    train_config, experiment_folder, pipeline_name, log_dir, fold_id,
    train_dataloader, valid_dataloader, binarizer_fn, eval_fn):
    
    fold_logger = init_logger(log_dir, 'train_fold_{}.log'.format(fold_id))

    best_checkpoint_folder = Path(experiment_folder, train_config['CHECKPOINTS']['BEST_FOLDER'])
    best_checkpoint_folder.mkdir(exist_ok=True, parents=True)

    checkpoints_history_folder = Path(
        experiment_folder,
        train_config['CHECKPOINTS']['FULL_FOLDER'],
        'fold{}'.format(fold_id)
    )
    checkpoints_history_folder.mkdir(exist_ok=True, parents=True)
    checkpoints_topk = train_config['CHECKPOINTS']['TOPK']

    calculation_name = '{}_fold{}'.format(pipeline_name, fold_id)
    
    device = train_config['DEVICE']
    
    module = importlib.import_module(train_config['MODEL']['PY'])
    model_class = getattr(module, train_config['MODEL']['CLASS'])
    model = model_class(**train_config['MODEL']['ARGS'])

    pretrained_model_config = train_config['MODEL'].get('PRETRAINED', False)
    if pretrained_model_config: 
        loaded_pipeline_name = pretrained_model_config['PIPELINE_NAME']
        pretrained_model_path = Path(
            pretrained_model_config['PIPELINE_PATH'], 
            pretrained_model_config['CHECKPOINTS_FOLDER'],
            '{}_fold{}.pth'.format(loaded_pipeline_name, fold_id)
        ) 
        if pretrained_model_path.is_file():
            model.load_state_dict(torch.load(pretrained_model_path))
            fold_logger.info('load model from {}'.format(pretrained_model_path)) 

    if len(train_config['DEVICE_LIST']) > 1:
        model = torch.nn.DataParallel(model)
    
    module = importlib.import_module(train_config['CRITERION']['PY'])
    loss_class = getattr(module, train_config['CRITERION']['CLASS'])
    loss_fn = loss_class(**train_config['CRITERION']['ARGS'])
    
    optimizer_class = getattr(torch.optim, train_config['OPTIMIZER']['CLASS'])
    optimizer = optimizer_class(model.parameters(), **train_config['OPTIMIZER']['ARGS'])
    scheduler_class = getattr(torch.optim.lr_scheduler, train_config['SCHEDULER']['CLASS'])
    scheduler = scheduler_class(optimizer, **train_config['SCHEDULER']['ARGS'])
    
    n_epoches = train_config['EPOCHES']
    grad_clip = train_config['GRADIENT_CLIPPING']
    grad_accum = train_config['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = train_config['EARLY_STOPPING']
    validation_frequency = train_config.get('VALIDATION_FREQUENCY', 1)
    
    freeze_model = train_config['MODEL']['FREEZE']

    # add IoU evaluation function
    eval_module_iou = importlib.import_module('Losses')
    eval_fn_iou = getattr(eval_module_iou, 'iou_metric')
    eval_fn_iou = functools.partial(eval_fn_iou, per_image=True)
    
    learning = Learning(
        optimizer,
        binarizer_fn,
        loss_fn,
        eval_fn,
        device,
        n_epoches,
        scheduler,
        freeze_model,
        grad_clip,
        grad_accum,
        early_stopping,
        validation_frequency,
        calculation_name,
        best_checkpoint_folder,
        checkpoints_history_folder,
        checkpoints_topk,
        fold_logger,
        eval_fn_iou
    )
    
    best_score, best_epoch, fold_time = learning.run_train(model,train_dataloader,valid_dataloader)
    
    return best_score, best_epoch, fold_time

def format_time(seconds):
    """add training time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def main():
    # print(os.listdir("./"))
    args = argparser()
    config_folder = Path(args.train_cfg.strip("/"))
    experiment_folder = config_folder.parents[0]

    train_config = load_yaml(config_folder)

    log_dir = Path(experiment_folder, train_config['LOGGER_DIR'])
    log_dir.mkdir(exist_ok=True, parents=True)

    main_logger = init_logger(log_dir, 'train_main.log')

    seed = train_config['SEED']
    init_seed(seed)
    main_logger.info(train_config)

    if "DEVICE_LIST" in train_config:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, train_config["DEVICE_LIST"]))

    pipeline_name = train_config['PIPELINE_NAME']
    dataset_folder = train_config['DATA_DIRECTORY'] 

    # Use simple resize transform instead of albumentations
    train_transform = SimpleResize(768, 768)
    valid_transform = SimpleResize(768, 768)

    non_empty_mask_proba = train_config.get('NON_EMPTY_MASK_PROBA', 0)
    use_sampler = train_config['USE_SAMPLER']

    dataset_folder = train_config['DATA_DIRECTORY'] 
    folds_distr_path = train_config['FOLD']['FILE'] 

    num_workers = train_config['WORKERS'] 
    batch_size = train_config['BATCH_SIZE'] 
    n_folds = train_config['FOLD']['NUMBER'] 

    usefolds = map(str, train_config['FOLD']['USEFOLDS'])
    # local_metric_fn, global_metric_fn = init_eval_fns(train_config)

    binarizer_module = importlib.import_module(train_config['MASK_BINARIZER']['PY'])
    binarizer_class = getattr(binarizer_module, train_config['MASK_BINARIZER']['CLASS'])
    binarizer_fn = binarizer_class(**train_config['MASK_BINARIZER']['ARGS'])

    eval_module = importlib.import_module(train_config['EVALUATION_METRIC']['PY'])
    eval_fn = getattr(eval_module, train_config['EVALUATION_METRIC']['CLASS'])
    eval_fn = functools.partial(eval_fn, **train_config['EVALUATION_METRIC']['ARGS'])

    # record started training time
    total_training_start = time.time()
    fold_times = []
    
    main_logger.info('=' * 60)
    main_logger.info('Training started at: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    main_logger.info('=' * 60)

    for fold_id in usefolds:
        main_logger.info('Start training of {} fold....'.format(fold_id))

        train_dataset = PneumothoraxDataset(
            data_folder=dataset_folder, mode='train', 
            transform=train_transform, fold_index=fold_id,
            folds_distr_path=folds_distr_path,
        )
        train_sampler = PneumoSampler(folds_distr_path, fold_id, non_empty_mask_proba)
        if use_sampler:
            train_dataloader =  DataLoader(
                dataset=train_dataset, batch_size=batch_size,   
                num_workers=num_workers, sampler=train_sampler, drop_last=True
            )
        else:
            train_dataloader =  DataLoader(
                dataset=train_dataset, batch_size=batch_size,   
                num_workers=num_workers, shuffle=True, drop_last=True
            )

        valid_dataset = PneumothoraxDataset(
            data_folder=dataset_folder, mode='val', 
            transform=valid_transform, fold_index=str(fold_id),
            folds_distr_path=folds_distr_path,
        )
        valid_dataloader =  DataLoader(
            dataset=valid_dataset, batch_size=batch_size, 
            num_workers=num_workers, shuffle=False
        )

        best_score, best_epoch, fold_time = train_fold(
            train_config, experiment_folder, pipeline_name, log_dir, fold_id,
            train_dataloader, valid_dataloader,
            binarizer_fn, eval_fn
        )
        fold_times.append(fold_time)
        
        main_logger.info('Fold {} completed - Best Score: {:.5f}, Best Epoch: {}, Time: {}'.format(
            fold_id, best_score, best_epoch, format_time(fold_time)))
    
    # record total training time
    total_training_time = time.time() - total_training_start
    avg_fold_time = sum(fold_times) / len(fold_times) if fold_times else 0
    
    main_logger.info('=' * 60)
    main_logger.info('All training completed at: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    main_logger.info('Total training time: {}'.format(format_time(total_training_time)))
    main_logger.info('Average fold time: {}'.format(format_time(avg_fold_time)))
    main_logger.info('Individual fold times:')
    for i, fold_time in enumerate(fold_times):
        main_logger.info('  Fold {}: {}'.format(i, format_time(fold_time)))
    main_logger.info('=' * 60)


if __name__ == "__main__":
    main()
