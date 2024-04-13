_base_ = [
    '../../_base_/default_runtime.py',
]

# 参照ReSSL设置

work_dir = './work_dirs/tiny/byol/byol_resnet18_1xb256-coslr-200e_tinyimagenet_diff-mlp-sgd-3x0cos-0m/linear_coslr'
checkpoint = './work_dirs/tiny/byol/byol_resnet18_1xb256-coslr-200e_tinyimagenet_diff-mlp-sgd-3x0cos-0m/epoch_200.pth'

auto_scale_lr = dict(base_batch_size=256, enable=True)
optim_wrapper = dict(
    optimizer=dict(lr=0.2, momentum=0.9, type='SGD', weight_decay=0.0),
    type='OptimWrapper')
param_scheduler = dict(
    T_max=100, begin=0, by_epoch=True, end=100, type='CosineAnnealingLR')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
val_cfg = dict()
test_cfg = dict()

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
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=200,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

data_root='/test2/datasets/imagenet/tiny-imagenet-200_format'
# data_root = '/home/wangbei/datasets/imagenet/tiny-imagenet-200_format'
# data_root = '/home/gao2/disk/datasets/imagenet'
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=200,
    mean=[255*0.4802, 255*0.4481, 255*0.3975],
    std=[255*.2302, 255*0.2265, 255*0.2262],
    to_rgb=True,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=64, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=256,
    num_workers=13,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        data_prefix='train',
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=256,
    num_workers=13,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        data_prefix='val',
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = val_dataloader
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50),

    )


