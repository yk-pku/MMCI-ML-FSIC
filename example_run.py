import os
import random
import argparse


def main():
    parser = argparse.ArgumentParser(description='multi-lable few-shot learning')
    parser.add_argument('cfg', default='configs/base.yaml', type=str)
    args = parser.parse_args()
    cfg = args.cfg
    print('cfg', cfg)
   
    tag = cfg.replace('.yaml', '')
    gpu_id = '0,1'

    num_gpus = len(gpu_id.split(','))
    port = random.randint(20000, 30000)
    # num_gpus = 8 if ntasks > 8 else ntasks
  
    test = ''

    if not os.path.exists('log'):
        os.mkdir('log')
    command = (
        'now=$(date +"%Y%m%d_%H%M%S")\n'
        'pwd=$(dirname $(readlink -f "$0"))\n'
        f'bk_dir=code_backup/{tag}_$now\n'
        'mkdir -p $bk_dir\n'
        'cp ./*.py $bk_dir\n'
        'cp -r datasets $bk_dir\n'
        'cp -r configs $bk_dir\n'
        'cp -r models $bk_dir\n'
        f'CUDA_VISIBLE_DEVICES={gpu_id} python -m torch.distributed.launch '
        f'--nproc_per_node={num_gpus} --master_port={port} '
        f'main.py --config $pwd/{cfg} '
        f'{test} '
        f'2>&1 | tee $pwd/log/{cfg[8 : -5]}.log-$now'
    )
    print(command)
    os.system(command)


if __name__ == '__main__':
    main()
