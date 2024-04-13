_base_ = [
    '../../_base_/datasets/imagenet_bs128_pil_resize_in100.py',
    '../../_base_/schedules/imagenet_sgd_coslr_100e.py',
    '../../_base_/default_runtime.py',
]

work_dir = ''
checkpoint = ''

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
train_dataloader = dict(batch_size=512, num_workers=8, persistent_workers=True, pin_memory=True)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.),
)

# learning rate scheduler
param_scheduler = [
   dict(type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=10)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))