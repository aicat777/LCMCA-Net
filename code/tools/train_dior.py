
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector on DIOR dataset')
    
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    
    parser.add_argument('--local-rank', type=int, default=0,
                       help='local rank for distributed training')
    
    
    parser.add_argument('--data-root', type=str, default='/mnt/cs_dsns_public/share/group_lvxiaodan/dior',
                       help='DIOR dataset root directory')
    parser.add_argument('--num-classes', type=int, default=20,
                       help='Number of classes in DIOR dataset (default: 20)')
    
    args = parser.parse_args()
    
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def setup_dior_config(cfg, args):
    """Setup DIOR dataset specific configurations"""
    
    if cfg.data.train.get('type') == 'CocoDataset':
        cfg.data.train.ann_file = osp.join(args.data_root, 'annotations/train.json')
        cfg.data.train.img_prefix = osp.join(args.data_root, './')
    
    if cfg.data.val.get('type') == 'CocoDataset':
        cfg.data.val.ann_file = osp.join(args.data_root, 'annotations/val.json')
        cfg.data.val.img_prefix = osp.join(args.data_root, './')
    if cfg.data.test.get('type') == 'CocoDataset':
        cfg.data.test.ann_file = osp.join(args.data_root, 'annotations/val.json')
        cfg.data.test.img_prefix = osp.join(args.data_root, './')
    
    
    if cfg.model.get('bbox_head'):
        if 'num_classes' in cfg.model.bbox_head:
            cfg.model.bbox_head.num_classes = args.num_classes
        
        for pipeline in cfg.data.train.pipeline:
            if pipeline.get('type') == 'LoadRPDV2Annotations':
                pipeline['num_classes'] = args.num_classes
    
    
    if args.launcher != 'none' or (args.gpus is not None and args.gpus > 1):
        
        original_batch_size = cfg.data.get('samples_per_gpu', 2)
        
        cfg.data.samples_per_gpu = original_batch_size
    
    return cfg


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    
    cfg = setup_dior_config(cfg, args)
    
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        config_name = osp.splitext(osp.basename(args.config))[0]
        cfg.work_dir = osp.join('./work_dirs', 'dior', config_name)
    
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    elif args.gpus is not None and args.launcher == 'none':
        
        cfg.gpu_ids = range(args.gpus)
    else:
        
        cfg.gpu_ids = [args.local_rank] if args.launcher != 'none' else [0]

    
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            print(f"Using DataParallel on GPUs: {cfg.gpu_ids}")
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        
        _, world_size = get_dist_info()
        cfg.gpu_ids = list(range(world_size))
        print(f"Distributed training with {world_size} GPUs")
        
        
        torch.cuda.set_device(args.local_rank)

    
    rank, _ = get_dist_info() if distributed else (0, 1)
    if rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    
    if distributed:
        torch.distributed.barrier()
    
    
    if rank == 0:
        config_dump_path = osp.join(cfg.work_dir, osp.basename(args.config))
        cfg.dump(config_dump_path)
    
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log') if rank == 0 else None
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    
    
    meta = dict()
    
    
    if rank == 0:
        
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        
        
        logger.info(f'DIOR Dataset Root: {args.data_root}')
        logger.info(f'Number of Classes: {args.num_classes}')
        logger.info(f'Using GPUs: {cfg.gpu_ids}')
        logger.info(f'Local Rank: {args.local_rank}')
        
        meta['config'] = cfg.pretty_text
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

    
    if args.seed is not None:
        if rank == 0:
            logger.info(f'Set random seed to {args.seed}, '
                        f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    
    datasets = [build_dataset(cfg.data.train)]
    
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    
    if cfg.checkpoint_config is not None and rank == 0:
        
        
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    
    
    model.CLASSES = datasets[0].CLASSES
    
    
    if distributed and cfg.optimizer.get('lr'):
        _, world_size = get_dist_info()
        base_lr = cfg.optimizer.lr
        
        
        samples_per_gpu = cfg.data.get('samples_per_gpu', 2) 
        
        
        current_total_bs = world_size * samples_per_gpu
        
        
        
        base_total_bs = 4 
        
        
        scaled_lr = base_lr * (current_total_bs / base_total_bs)
        
        cfg.optimizer.lr = scaled_lr
        if rank == 0:
            logger.info(f"Linear Scaling LR: base_lr ({base_lr}) scaled to {scaled_lr:.6f}. "
                        f"(Total BS changed from {base_total_bs} to {current_total_bs} "
                        f"-> GPUs: {world_size}, BS/GPU: {samples_per_gpu})")
    
    
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()