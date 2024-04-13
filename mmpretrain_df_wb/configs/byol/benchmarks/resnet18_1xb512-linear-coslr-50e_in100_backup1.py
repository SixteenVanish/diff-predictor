_base_ = [
    '../../_base_/default_runtime.py',
]

work_dir = './work_dirs/tmp/tmp2'
checkpoint = './work_dirs/in100/byol/byol_resnet18_1xb512-coslr-100e_in100_diff-mlp-sgd-2x0cos-uniformT/epoch_100.pth'
# load_from = './work_dirs/in100/byol/byol_resnet18_1xb512-coslr-100e_in100_diff-mlp-sgd-2x0cos-uniformT/linear/epoch_26.pth'
# resume = True

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
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset settings
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=100,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

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
    batch_size=512,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='/home/wangbei/datasets/imagenet/',
        ann_file='/home/wangbei/datasets/imagenet/train.txt_100_withlabel.txt',
        with_label=True,
        #split='',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/home/wangbei/datasets/imagenet',
        ann_file='/home/wangbei/datasets/imagenet/val.txt_100_withlabel.txt',
        with_label=True,
        #split='',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# train_dataloader = dict(
#     batch_size=512,
#     num_workers=8,
#     dataset=dict(
#         type='CustomDataset',
#         data_root='/home/wangbei/datasets/imagenet/imagenet-100',
#         data_prefix='train',
#         pipeline=train_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=True),
# )
# val_dataloader = dict(
#     batch_size=128,
#     num_workers=5,
#     dataset=dict(
#         type='CustomDataset',
#         data_root='/home/wangbei/datasets/imagenet/imagenet-100',
#         data_prefix='val',
#         pipeline=test_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=False),
# )

# train_dataloader = dict(
#     batch_size=128,
#     num_workers=8,
#     dataset=dict(
#         type='CustomDataset',
#         data_root='/home/wangbei/datasets/imagenet',
#         data_prefix='train',
#         pipeline=train_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=True),
# )
# val_dataloader = dict(
#     batch_size=128,
#     num_workers=5,
#     dataset=dict(
#         type='CustomDataset',
#         data_root='/home/wangbei/datasets/imagenet',
#         data_prefix='val',
#         pipeline=test_pipeline),
#     sampler=dict(type='DefaultSampler', shuffle=False),
# )

val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=10)
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