_base_ = [
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../_base_/default_runtime.py'
]

# 参考mae
checkpoint = "../diff-ijepa/logs/vitb16.224-8xb256-100e_in1k/jepa-ep100_.pth.tar"
work_dir = "./work_dirs/in1k/ijepa/vitb16.224-8xb256-100e_in1k/linear_followMAE"

# dataset settings
data_root = "/ssd/datasets/imagenet"
train_dataloader = dict(batch_size=256, drop_last=True, dataset=dict(data_root=data_root))
val_dataloader = dict(drop_last=False, dataset=dict(data_root=data_root))
test_dataloader = dict(drop_last=False, dataset=dict(data_root=data_root))

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ijepa_vit',
        arch='base',
        img_size=[224],
        patch_size=16,
        frozen_stages=12,
        out_type='avg_featmap',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='target_encoder.')),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)]),
    )

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='LARS', lr=6.4, weight_decay=0.0, momentum=0.9))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        by_epoch=True,
        begin=10,
        end=90,
        eta_min=0.0,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=90)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=100))

randomness = dict(seed=0, diff_rank_seed=True)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=5),
    dist_cfg=dict(backend='nccl'))