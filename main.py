import os
import re
import argparse
import random
import time
import yaml
import json
import logging
from addict import Dict
import numpy as np
import torch
from torch.backends import cudnn
import torch.distributed as dist

import train_eval
import models

def dist_init():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    host_ip = os.environ['MASTER_ADDR']
    port = os.environ['MASTER_PORT']
    init_method = 'tcp://{}:{}'.format(host_ip, port)
    print('dist.init_process_group', init_method, world_size, rank)
    dist.init_process_group('nccl', init_method=init_method, world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    print('rank is {}, local_rank is {}, world_size is {}, host ip is {}'.format(rank, local_rank, world_size, host_ip))
    return local_rank


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_configs(configs, base_configs):
    for key in base_configs:
        if not key in configs:
            configs[key] = base_configs[key]
        elif type(configs[key]) is dict:
            merge_configs(configs[key], base_configs[key])


def build_configs(config_file, loaded_config_files):
    loaded_config_files.append(config_file)
    with open(config_file, 'r') as reader:
        configs = yaml.load(reader, Loader=yaml.Loader)
    for base_config_file in configs['base']:
        base_config_file = os.getcwd() + '/configs/' + base_config_file
        if base_config_file in loaded_config_files:
            continue
        base_configs = build_configs(base_config_file, loaded_config_files)
        merge_configs(configs, base_configs)
    return configs


def clear_configs(configs):
    keys = list(configs.keys())
    for key in keys:
        if type(configs[key]) is dict:
            configs[key] = clear_configs(configs[key])
        elif configs[key] == 'None':
            print('Clear config', key)
            configs.pop(key)
    return configs


def main():
    # Get parser
    parser = argparse.ArgumentParser(description='multi-lable few-shot learning')
    parser.add_argument('--config_file', default='configs/base.yaml', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--local_rank', type=int)
    # Get configs
    args = parser.parse_args()
    out_dir = args.config_file.replace('configs', 'output')
    out_dir = out_dir.replace('.yaml', '')
    tag = args.config_file.split('/')[-1].split('.')[0]
    configs = Dict(clear_configs(build_configs(args.config_file, [])))
    # Basic configuration
    print(f'Seed {configs.seed}')
    seed_all(configs.seed)
    local_rank = dist_init()
    # local_rank = dist_init(configs.port)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    configs.rank = rank
    configs.local_rank = local_rank
    configs.world_size = world_size
    cudnn.benchmark = False
    cudnn.deterministic = True
    # Get logger and logger_all
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    logger_all = logging.getLogger('all')
    # Set loggers
    if dist.get_rank() == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    logger_all.setLevel(logging.INFO)
    # Make saving directories
    configs.out_dir = out_dir
    ckpt_dir = configs.out_dir + '/checkpoints'
    infer_dir = configs.out_dir + '/infer_results'
    metric_dir = configs.out_dir + '/' + tag
    if dist.get_rank() == 0:
        if not os.path.exists(configs.out_dir):
            os.makedirs(configs.out_dir)
            logger.info(f'create output directory {configs.out_dir}')
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
            logger.info(f'create checkpoint directory {ckpt_dir}')
        if not os.path.exists(infer_dir):
            os.mkdir(infer_dir)
            logger.info(f'create inference directory {infer_dir}')
        if not os.path.exists(metric_dir):
            os.mkdir(metric_dir)
            logger.info(f'create metric directory {metric_dir}')
    configs.ckpt_dir = ckpt_dir
    configs.infer_dir = infer_dir
    configs.metric_dir = metric_dir
    configs.tag = tag
    logger.info(
        f'configs\n{json.dumps(configs, indent=2, ensure_ascii=False)}'
    )
    assert torch.cuda.is_available()
    configs.device = torch.device('cuda')
    logger.info(f'Device is {configs.device}')
    # Get model
    logger.info(f'creating model: {configs.model.kind}')
    model = models.__dict__[configs.model.kind](**configs.model.kwargs)
    # Synchronize batch norm layers
    assert type(configs.model.sync_bn) is bool
    if configs.model.sync_bn:
        logger.info('Use SyncBatchNorm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    # Get optimizer
    optimizer = torch.optim.__dict__[configs.trainer.optimizer.kind](
        filter(lambda x: x.requires_grad is not False, model.parameters()), **configs.trainer.optimizer.kwargs
    )
    logger.info(f'optimizer:\n{optimizer}')
    # Epoch setting
    configs.max_epoch = configs.trainer.max_epoch
    # Resume model, optimizer
    if configs.model.resume:
        logger.info('Resuming model')
        # logger.info(f'Before {model.state_dict()}')
        resume_path = configs.model.resume
        if not os.path.isfile(resume_path):
            raise Exception(
                f'Not found resume model: {resume_path}'
            )
        checkpoint = torch.load(resume_path)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['state_dict'], strict=False
        )
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(
            f'Finish resuming from {configs.model.resume} '
            f'Missing keys {missing_keys}, Unexpected keys {unexpected_keys}'
        )
        # logger.info(f'After {model.state_dict()}')
        del checkpoint
    # Distribute model
    model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True,
        device_ids=[configs.local_rank],
    )
    logger.info(f'Use model:\n{model}')
    # Jump to specific task
    train_eval.main(
        configs=configs, is_test=args.test, model=model,
        optimizer=optimizer,
        logger=logger, logger_all=logger_all
    )


if __name__ == '__main__':
    main()
