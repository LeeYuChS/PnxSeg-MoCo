import torch
from torch.nn.utils import clip_grad_norm_
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
import numpy as np
import time
import datetime

from tqdm import tqdm
from pathlib import Path

import heapq
from collections import defaultdict

class Learning():
    def __init__(self,
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
                 logger,
                 eval_fn_iou=None  # add IoU evaluation function
        ):
        self.logger = logger

        self.optimizer = optimizer
        self.binarizer_fn = binarizer_fn
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.eval_fn_iou = eval_fn_iou

        self.device = device
        self.n_epoches = n_epoches
        self.scheduler = scheduler
        self.freeze_model = freeze_model
        self.grad_clip = grad_clip
        self.grad_accum = grad_accum
        self.early_stopping = early_stopping
        self.validation_frequency = validation_frequency

        # time tracking - only record fold level
        self.fold_start_time = None

        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder, 
            '{}.pth'.format(self.calculation_name)
        )
        self.checkpoints_history_folder = Path(checkpoints_history_folder)
        self.checkpoints_history_folder.mkdir(exist_ok=True, parents=True)
        
        self.checkpoints_topk = checkpoints_topk
        self.score_heap = []
        # csv name with calculation_name
        self.summary_file = Path(self.checkpoints_history_folder, f'{self.calculation_name}_training_summary.csv')     
        if self.summary_file.is_file():
            try:
                summary_df = pd.read_csv(self.summary_file)
                # prefer known column names, fall back to heuristics
                possible_cols = [c for c in ['best_metric', 'best_dice', 'best_score', 'score', 'best_iou'] if c in summary_df.columns]
                if not possible_cols:
                    possible_cols = [col for col in summary_df.columns if any(k in col.lower() for k in ('best', 'dice', 'score'))]

                if possible_cols:
                    chosen = possible_cols[0]
                    try:
                        self.best_score = float(summary_df[chosen].max())
                    except Exception:
                        self.best_score = 0.0

                    # try infer best epoch index
                    self.best_epoch = -1
                    if 'epoch' in summary_df.columns:
                        try:
                            idx = summary_df[chosen].idxmax()
                            self.best_epoch = int(summary_df.loc[idx, 'epoch'])
                        except Exception:
                            self.best_epoch = -1

                    logger.info('Pretrained best score (from "{}") is {:.5f}'.format(chosen, self.best_score))
                else:
                    logger.warning('summary.csv found but no recognizable score column; starting from zero')
                    self.best_score = 0.0
                    self.best_epoch = -1
            except Exception as e:
                logger.warning('Failed to read summary.csv: {}'.format(e))
                self.best_score = 0.0
                self.best_epoch = -1
        else:
            self.best_score = 0.0
            self.best_epoch = -1

    def train_epoch(self, model, loader):
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0

        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):
            loss, predicted = self.batch_train(model, imgs, labels, batch_idx)

            # just slide average
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)

            tqdm_loader.set_description('loss: {:.4f} lr:{:.6f}'.format(
                current_loss_mean, self.optimizer.param_groups[0]['lr']))
        return current_loss_mean

    def batch_train(self, model, batch_imgs, batch_labels, batch_idx):
        batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
        predicted = model(batch_imgs)
        loss = self.loss_fn(predicted, batch_labels)

        loss.backward()
        if batch_idx % self.grad_accum == self.grad_accum - 1:
            clip_grad_norm_(model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item(), predicted

    def valid_epoch(self, model, loader):
        tqdm_loader = tqdm(loader)
        current_score_mean = 0
        used_thresholds = self.binarizer_fn.thresholds
        metrics = defaultdict(float)
        iou_metrics = defaultdict(float)  # add IoU metric
        val_loss_sum = 0  # add validation loss

        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):
            with torch.no_grad():
                predicted_probas = self.batch_valid(model, imgs)
                labels = labels.to(self.device)
                
                # calculate validation loss
                val_loss = self.loss_fn(predicted_probas, labels).item()
                val_loss_sum = (val_loss_sum * batch_idx + val_loss) / (batch_idx + 1)
                
                mask_generator = self.binarizer_fn.transform(predicted_probas)
                for current_thr, current_mask in zip(used_thresholds, mask_generator):
                    # calculate Dice score
                    current_metric = self.eval_fn(current_mask, labels).item()
                    current_thr = tuple(current_thr)
                    metrics[current_thr] = (metrics[current_thr] * batch_idx + current_metric) / (batch_idx + 1)

                    # calculate IoU score (if IoU evaluation function is provided)
                    if self.eval_fn_iou is not None:
                        current_iou = self.eval_fn_iou(current_mask, labels).item()
                        iou_metrics[current_thr] = (iou_metrics[current_thr] * batch_idx + current_iou) / (batch_idx + 1)

                best_threshold = max(metrics, key=metrics.get)
                best_metric = metrics[best_threshold]
                tqdm_loader.set_description('Dice: {:.5f}, Loss: {:.5f} on {}'.format(best_metric, val_loss_sum, best_threshold))

        return metrics, iou_metrics, val_loss_sum, best_metric

    def batch_valid(self, model, batch_imgs):
        batch_imgs = batch_imgs.to(self.device)
        
        # Model should already be in eval mode during validation
        # No need to handle training mode since valid_epoch calls model.eval() beforehand
        predicted = model(batch_imgs)
        predicted = torch.sigmoid(predicted)
        return predicted

    def process_summary(self, metrics, iou_metrics, val_loss, train_loss, epoch):
        best_threshold = max(metrics, key=metrics.get)
        best_metric = metrics[best_threshold]
        best_iou = iou_metrics[best_threshold] if self.eval_fn_iou is not None else 0.0

        # create epoch summary with all metrics
        epoch_summary = pd.DataFrame.from_dict([metrics])
        epoch_summary['epoch'] = epoch
        epoch_summary['best_dice'] = best_metric
        epoch_summary['best_iou'] = best_iou
        epoch_summary['val_loss'] = val_loss
        epoch_summary['train_loss'] = train_loss

        # add IoU metric columns
        if self.eval_fn_iou is not None:
            for thr, iou_val in iou_metrics.items():
                epoch_summary[f'iou_{str(thr)}'] = iou_val

        # reorder columns
        base_cols = ['epoch', 'best_dice', 'best_iou', 'val_loss', 'train_loss']
        other_cols = [col for col in epoch_summary.columns if col not in base_cols]
        epoch_summary = epoch_summary[base_cols + other_cols]
        epoch_summary.columns = [str(col) for col in epoch_summary.columns]

        self.logger.info('{} epoch: \t Dice: {:.5f}, IoU: {:.5f}, Val Loss: {:.5f}, Train Loss: {:.5f}\t Params: {}'.format(
            epoch, best_metric, best_iou, val_loss, train_loss, best_threshold))

        # csv save with error handling
        try:
            if not self.summary_file.is_file():
                epoch_summary.to_csv(self.summary_file, index=False)
            else:
                summary = pd.read_csv(self.summary_file)
                summary = pd.concat([summary, epoch_summary], ignore_index=True)
                summary.to_csv(self.summary_file, index=False)
        except Exception as e:
            self.logger.error(f'Failed to save summary: {e}')  

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    def format_time(self, seconds):
        """format time display"""
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

    def post_processing(self, score, epoch, model):
        if self.freeze_model:
            return

        checkpoints_history_path = Path(
            self.checkpoints_history_folder, 
            '{}_epoch{}.pth'.format(self.calculation_name, epoch)
        )

        torch.save(self.get_state_dict(model), checkpoints_history_path)
        heapq.heappush(self.score_heap, (score, checkpoints_history_path))
        if len(self.score_heap) > self.checkpoints_topk:
            _, removing_checkpoint_path = heapq.heappop(self.score_heap)
            removing_checkpoint_path.unlink()
            self.logger.info('Removed checkpoint is {}'.format(removing_checkpoint_path))
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            torch.save(self.get_state_dict(model), self.best_checkpoint_path)
            self.logger.info('best model: {} epoch - {:.5f}'.format(epoch, score))

        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def run_train(self, model, train_dataloader, valid_dataloader):
        # start time tracking - only record fold level
        self.fold_start_time = time.time()
        model.to(self.device)
        
        self.logger.info('Starting fold training at: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        for epoch in range(self.n_epoches):
            train_loss_mean = 0
            
            if not self.freeze_model:
                self.logger.info('{} epoch: \t start training....'.format(epoch))
                model.train()
                train_loss_mean = self.train_epoch(model, train_dataloader)
                self.logger.info('{} epoch: \t Calculated train loss: {:.5f}'.format(epoch, train_loss_mean))

            if epoch % self.validation_frequency != (self.validation_frequency - 1):
                self.logger.info('skip validation....')
                continue

            self.logger.info('{} epoch: \t start validation....'.format(epoch))
            model.eval()
            metrics, iou_metrics, val_loss, score = self.valid_epoch(model, valid_dataloader)

            self.process_summary(metrics, iou_metrics, val_loss, train_loss_mean, epoch)

            self.post_processing(score, epoch, model)

            if epoch - self.best_epoch > self.early_stopping:
                self.logger.info('EARLY STOPPING')
                break

        # record fold total training time
        fold_total_time = time.time() - self.fold_start_time
        
        self.logger.info('Fold training completed at: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.logger.info('Total fold training time: {}'.format(self.format_time(fold_total_time)))
        
        return self.best_score, self.best_epoch, fold_total_time
        
