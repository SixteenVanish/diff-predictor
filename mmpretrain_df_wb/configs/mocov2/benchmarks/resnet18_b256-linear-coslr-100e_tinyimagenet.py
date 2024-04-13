_base_ = [
    '../../../../_base_/default_runtime.py',
]

auto_scale_lr = dict(base_batch_size=256)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=4,
        init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=200,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# dataset settings
dataset_type = 'CustomDataset'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=64, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=64, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=64),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='/home/wangbei/datasets/imagenet/tiny-imagenet-200_format',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=256,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='/home/wangbei/datasets/imagenet/tiny-imagenet-200_format',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.))
# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100)
# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# runtime settings
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

data_preprocessor = dict(
    num_classes=200,
    mean=[255*0.4802, 255*0.4481, 255*0.3975],
    std=[255*0.2302, 255*0.2265, 255*0.2262],
    to_rgb=True)
