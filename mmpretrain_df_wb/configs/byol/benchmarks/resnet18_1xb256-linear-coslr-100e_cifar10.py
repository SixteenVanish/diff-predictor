_base_ = [
    '../../_base_/default_runtime.py',
]

# 参照ReSSL设置

work_dir = './work_dirs/cifar10/byol/byol_resnet18_1xb256-coslr-500e_cifar10_diff-mlp-sgd-1cos-1x0cos/linear_coslr'
checkpoint = './work_dirs/cifar10/byol/byol_resnet18_1xb256-coslr-500e_cifar10_diff-mlp-sgd-1cos-1x0cos/epoch_500.pth'

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
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# data_root='/test2/datasets/cifar'
data_root = '/home/wangbei/datasets/cifar'
# data_root = '/home/gao2/disk/datasets/cifar'
dataset_type = 'CIFAR10'
data_preprocessor = dict(
    num_classes=10,
    mean=[255*0.4914, 255*0.4822, 255*0.4465],
    std=[255*0.2023, 255*0.1994, 255*0.2010],
    to_rgb=True,
)
train_pipeline = [
    dict(type='RandomResizedCrop', scale=64, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        download=False,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        download=False,
        split='test',
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = val_dataloader
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=40),
)
