
dataset_type = 'CocoDataset'  

# DIOR数据集的20个类别
classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 
    'bridge', 'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
    'tenniscourt', 'trainstation', 'vehicle', 'windmill'
)

# 数据预处理
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# 训练流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='LoadRPDV2Annotations', num_classes=20),  
    dict(type='RPDV2FormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_sem_map', 'gt_sem_weights']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=10, 
    workers_per_gpu=8, 
    
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=None, 
        img_prefix=None,
        pipeline=train_pipeline),
    
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=None,
        img_prefix=None,
        pipeline=test_pipeline),
    
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=None,
        img_prefix=None,
        pipeline=test_pipeline)
)