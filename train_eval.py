import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
# import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil

import datasets

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(
        phase, epoch, configs, data_loader, model, optimizer,
        lr_schedule, metric, logger, logger_all
    ):
    end = time.time()
    total_scores = {}
    logger.info(f'Epoch {epoch} with {len(data_loader)} iterations')
    if phase != 'train':
        all_pred_scores = []
        all_gt_labels = []
    # Collect all of the predicting labels
    for b_ind, (images, gt_labels, indexs, lstm_labels, label_numbers) in enumerate(data_loader):
        images = images.cuda(non_blocking = True)
        gt_labels = gt_labels.cuda(non_blocking = True)

        loss_names, loss_items, total_loss = model(images, gt_labels)
        
        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # record losses
        scores = {}
        for loss_name, loss_item in zip(loss_names, loss_items):
            scores[loss_name] = loss_item

        for key, value in scores.items():
            if not key in total_scores:
                total_scores[key] = []
            total_scores[key].append(value)

        if configs.device.type == 'cuda':
            memory = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3)
        else:
            memory = 0.

        time_cost = time.time() - end
        logger_all.info(
            f'{configs.tag} {phase} '
            f'Rank {dist.get_rank()}/{dist.get_world_size()} '
            f'Epoch {epoch}/{configs.max_epoch} '
            f'Batch {b_ind}/{len(data_loader)} '
            f'Time {time_cost: .3f} Mem {memory}MB '
            f'LR {optimizer.param_groups[0]["lr"]:.3e} '
            f'{scores}'
        )
   
    
    total_scores = dict(
        [
            (key, round(np.mean(value), 4))
            for key, value in total_scores.items()
        ]
    )
    return total_scores

def main(configs, is_test, model, optimizer, logger, logger_all, **kwargs):
    # Get metric function
    metric = Metric(**configs.metric)
    # Build dataloader
    d_kind = configs.dataset.k_kind if 'k_kind' in configs.dataset else configs.dataset.kind
    train_dataset = datasets.__dict__[d_kind](**configs.dataset.kwargs)
    logger.info(train_dataset.info)
    shuffle = True if 'shuffle' not in configs.dataset else configs.dataset.shuffle
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=shuffle
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=configs.dataset.train_batch_size,
        num_workers=configs.dataset.num_workers,
        pin_memory=False, drop_last=True,
        worker_init_fn=seed_worker
    )
    logger.info(f'{len(train_dataset)} train images')
    lr_configs = configs.trainer.lr_schedule
    warmup_lr_schedule = np.linspace(
        lr_configs.start_warmup, lr_configs.base_lr,
        len(train_loader) * lr_configs.warmup_epochs
    )
    iters = np.arange(len(train_loader) * lr_configs.cosine_epochs)
    cosine_lr_schedule = np.array(
        [
            lr_configs.final_lr + (
                0.5 * (lr_configs.base_lr - lr_configs.final_lr)
                * (1 + math.cos(math.pi * t / (len(train_loader) * lr_configs.cosine_epochs)))
            )
            for t in iters
        ]
    )
    lr_schedule = np.concatenate([warmup_lr_schedule] + [cosine_lr_schedule] * lr_configs.cosine_times)
    plt.plot(range(len(lr_schedule)), lr_schedule)
    plt.xlabel('iterations')
    plt.ylabel('learning rate')
    configs.max_epoch = lr_configs.warmup_epochs + lr_configs.cosine_epochs * lr_configs.cosine_times

    for epoch in range(configs.max_epoch):
        dist.barrier()
        train_sampler.set_epoch(epoch)
        model.train()

        score = run_epoch(
            phase='train', epoch=epoch, configs=configs, data_loader=train_loader,
            model=model, optimizer=optimizer,
            lr_schedule=lr_schedule, metric=metric,
            logger=logger, logger_all=logger_all,
        )
        
        logger_all.info(
            f'Rank {dist.get_rank()}/{dist.get_world_size()} '
            f'Train Epoch {epoch} {score}'
        )
        if (
            (dist.get_rank() == 0) and (
                ((epoch + 1) % configs.trainer.save_freq == 0)
                or (epoch == configs.max_epoch - 1)
            )
        ):
            state_dict = model.module.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model': configs.model.kind,
                'score': score,
                'state_dict': state_dict,
            }
            ckpt_path = (
                f'{configs.ckpt_dir}/{configs.model.kind}_epoch_{epoch}.pth'
            )
            torch.save(checkpoint, ckpt_path)
            os.system(f'cp {configs.ckpt_dir}/{configs.model.kind}_epoch_{epoch}.pth {configs.ckpt_dir}/latest.pth')
            logger.info(f'Save checkpoint in epoch {epoch}')
        
        # To avoid deadlock
        dist.barrier()
        
        time.sleep(2.33)