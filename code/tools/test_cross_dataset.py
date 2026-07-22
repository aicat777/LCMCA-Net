# tools/test_cross_dataset.py
import argparse
import os
import json
import numpy as np

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

# ==================== 类别定义与映射 ====================
DIOR_CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt',
    'bridge', 'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
    'tenniscourt', 'trainstation', 'vehicle', 'windmill'
)
DIOR_NAME2ID = {name: i for i, name in enumerate(DIOR_CLASSES)}

NWPU_CLASSES = (
    'airplane', 'ship', 'storage_tank', 'baseball_diamond',
    'tennis_court', 'basketball_court', 'ground_track_field',
    'harbor', 'bridge', 'vehicle'
)

RSOD_EVAL_CLASSES = ('aircraft', 'oiltank', 'overpass')

# 名称 -> ID 映射（用于构造源类ID到目标类ID）
DIOR2NWPU = {
    'airplane': 0,
    'ship': 1,
    'storagetank': 2,
    'baseballfield': 3,
    'tenniscourt': 4,
    'basketballcourt': 5,
    'groundtrackfield': 6,
    'harbor': 7,
    'bridge': 8,
    'vehicle': 9,
}

DIOR2RSOD = {
    'airplane': 0,
    'storagetank': 1,
    'overpass': 2,
}


def remap_detections_mmdet(det_results, src_id_to_tgt_id, num_tgt_classes):
    """
    对 MMDetection 标准输出进行类别重映射。
    det_results: list of list of np.ndarray,
                 每张图是一个长度为源类别数的列表，
                 每个元素是形状 (N,5) 的数组（x1,y1,x2,y2,score）。
    src_id_to_tgt_id: dict, {源类别id: 目标类别id}
    num_tgt_classes: int, 目标数据集类别数
    返回: 相同结构，但每张图列表长度变为 num_tgt_classes。
    """
    new_results = []
    for img_dets in det_results:
        # 初始化目标类别容器，全为空数组
        tgt_img_dets = [np.zeros((0, 5), dtype=np.float32) for _ in range(num_tgt_classes)]
        for src_cls, tgt_cls in src_id_to_tgt_id.items():
            if src_cls < len(img_dets) and len(img_dets[src_cls]) > 0:
                # 合并到目标类别（若以后有多个源类映射到同一目标类，这里也可正常工作）
                tgt_img_dets[tgt_cls] = np.concatenate(
                    [tgt_img_dets[tgt_cls], img_dets[src_cls]], axis=0)
        new_results.append(tgt_img_dets)
    return new_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cross-dataset evaluation: DIOR trained model on NWPU VHR-10 / RSOD')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--data-root', type=str, required=True,
        help='Root directory of target dataset')
    parser.add_argument(
        '--target-dataset', type=str, required=True, choices=['nwpu', 'rsod'],
        help='Target dataset for cross evaluation')
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--format-only', action='store_true')
    parser.add_argument('--eval', type=str, nargs='+', default=['bbox'])
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-dir', help='directory for saving visualized images')
    parser.add_argument('--show-score-thr', type=float, default=0.3)
    parser.add_argument('--gpu-collect', action='store_true')
    parser.add_argument('--tmpdir', help='tmp directory for collecting results')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--eval-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def setup_cross_config(cfg, args):
    """根据目标数据集修改测试配置"""
    if args.target_dataset == 'nwpu':
        cfg.data.test.ann_file = os.path.join(args.data_root, 'instances_val2017.json')
        cfg.data.test.img_prefix = os.path.join(args.data_root, 'image')
        cfg.data.test.classes = NWPU_CLASSES
    elif args.target_dataset == 'rsod':
        cfg.data.test.ann_file = os.path.join(args.data_root, 'Annotations/RSOD_test_eval3.json')
        cfg.data.test.img_prefix = os.path.join(args.data_root, 'JPEGImages/')
        cfg.data.test.classes = RSOD_EVAL_CLASSES
    cfg.data.test.test_mode = True
    return cfg


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, \
        'Please specify at least one operation (save/eval/format/show)'

    cfg = Config.fromfile(args.config)
    cfg = setup_cross_config(cfg, args)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone') and neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone') and cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test)
    # 目标数据集类别
    if args.target_dataset == 'nwpu':
        dataset.CLASSES = NWPU_CLASSES
    else:
        dataset.CLASSES = RSOD_EVAL_CLASSES

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # 保持模型内部类别数为 DIOR 的 20 类
    model.CLASSES = DIOR_CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    # ================== 类别重映射 ==================
    if args.target_dataset == 'nwpu':
        src2tgt = {DIOR_NAME2ID[name]: nwpu_id for name, nwpu_id in DIOR2NWPU.items()}
        outputs = remap_detections_mmdet(outputs, src2tgt, len(NWPU_CLASSES))
        print('Detection results remapped to NWPU VHR-10 classes.')
    elif args.target_dataset == 'rsod':
        src2tgt = {DIOR_NAME2ID[name]: rsod_id for name, rsod_id in DIOR2RSOD.items()}
        outputs = remap_detections_mmdet(outputs, src2tgt, len(RSOD_EVAL_CLASSES))
        print('Detection results remapped to RSOD evaluation classes.')

    # ================== 评估 ==================
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nWriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            results = dataset.evaluate(outputs, **eval_kwargs)

            print("\n" + "="*60)
            print(f"Cross-dataset Evaluation: DIOR -> {args.target_dataset.upper()}")
            print("="*60)
            if 'bbox_mAP' in results:
                print(f"mAP@0.5:0.95: {results['bbox_mAP']:.4f}")
                print(f"mAP@0.5: {results['bbox_mAP_50']:.4f}")
                print(f"mAP@0.75: {results['bbox_mAP_75']:.4f}")
            if 'bbox_mAP_copypaste' in results:
                print("\nPer-class AP:")
                print(results['bbox_mAP_copypaste'])
            print("="*60)


if __name__ == '__main__':
    main()