_base_ = [
    '../../_base_/default_runtime.py',
]

# 参照ReSSL设置

work_dir = './work_dirs/byol_resnet50_2xb128-coslr-100e_tinyimagenet/linear'
checkpoint = './work_dirs/byol_resnet50_2xb128-coslr-100e_tinyimagenet/epoch_200.pth'

auto_scale_lr = dict(base_batch_size=256, enable=True)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=10, momentum=0.9, weight_decay=0.))
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[60, 80], gamma=0.1)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50)
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

# data_root='/test2/datasets/imagenet/tiny-imagenet-200_format'
data_root = '/home/wangbei/datasets/imagenet/tiny-imagenet-200_format'
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
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        data_prefix='train',
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=8,
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
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=80),
)
