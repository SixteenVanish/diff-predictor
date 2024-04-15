_base_ = [
    '../../_base_/datasets/imagenet_bs64_swin_224.py',
    '../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../_base_/default_runtime.py'
]

checkpoint = "../diff-ijepa/logs/vitb16.224-8xb256-100e_in1k/jepa-ep100_.pth.tar"
work_dir = "./work_dirs/in1k/ijepa/vitb16.224-8xb256-100e_in1k/linear_coslr"

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

data_root = "/ssd/datasets/imagenet"
train_dataloader = dict(batch_size=256, dataset=dict(pipeline=train_pipeline, data_root=data_root))
val_dataloader = dict(batch_size=256, dataset=dict(pipeline=test_pipeline, data_root=data_root))
test_dataloader = val_dataloader

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ijepa_vit',
        arch='base',
        img_size=[224],
        patch_size=16,
        frozen_stages=12,
        out_type='raw',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=2e-3, weight_decay=0.05, betas=(0.9, 0.999)),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.65,
        custom_keys={
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

train_cfg = dict(by_epoch=True, max_epochs=100)

randomness = dict(seed=0, diff_rank_seed=True)
