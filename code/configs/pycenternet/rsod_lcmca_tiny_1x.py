_base_ = [
    '../_base_/datasets/rsod_detection.py',              
    '../_base_/schedules/schedule_1x.py',  
    '../_base_/default_runtime.py'         
]

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='PyCenterNetDetector',
    pretrained=None,
    backbone=dict(
        type='LSKAConvNeXtv1',
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='PyCenterNetHead',
        num_classes=4,                     
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        shared_stacked_convs=1,
        first_kernel_size=3,
        kernel_size=1,
        corner_dim=64,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        conv_module_style='norm',
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='GIoULoss', loss_weight=1.0),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=0.25),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0/9.0, loss_weight=1.0),
        loss_sem=dict(
            type='SEPFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.1))
)

train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssignerV2', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    heatmap=dict(
        assigner=dict(type='PointHMAssignerV2', gaussian_bump=True, gaussian_iou=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))

test_cfg = dict(
    distance_threshold=0.5,
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2), _delete_=True)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)

evaluation = dict(
    interval=10,
    metric='bbox',
    iou_thrs=[0.5],
    classwise=True
)
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'


load_from = '/mnt/cs_dsns_public/cs_dsns_panwentao/code/od/code/work_dirs/backup/dior/lcv6_tiny_adamw_6x/epoch_66.pth'
resume_from = None
workflow = [('train', 1)]