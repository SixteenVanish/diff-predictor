_base_ = [
    '../../_base_/datasets/imagenet_bs128_pil_resize_in100.py',
    '../../_base_/default_runtime.py',
]

work_dir = './work_dirs/in-c10-2D/byol/byol_resnet18_1xb512-coslr-50e_in-c10-2D_mc1010_diff-mlp-sgd-3x0cos-1e-4noisemse-uniformT/linear'
checkpoint = './work_dirs/in-c10-2D/byol/byol_resnet18_1xb512-coslr-50e_in-c10-2D_mc1010_diff-mlp-sgd-3x0cos-1e-4noisemse-uniformT/epoch_50.pth'


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=4, 
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset settings
train_dataloader = dict(batch_size=512, num_workers=8, persistent_workers=True, pin_memory=True)
data_preprocessor = dict(
    num_classes=10,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
dataset_type = 'ImageNet'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/data/wangbei/datasets/imagenet/',
        ann_file='/mnt/data/wangbei/datasets/imagenet/train_10class.txt',
        with_label=True,
        split='',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/data/wangbei/datasets/imagenet',
        ann_file='/mnt/data/wangbei/datasets/imagenet/val_10class.txt',
        with_label=True,
        split='',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.),
)

# learning rate scheduler
param_scheduler = [
   dict(type='CosineAnnealingLR', T_max=50, by_epoch=True, begin=0, end=50)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=100),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))