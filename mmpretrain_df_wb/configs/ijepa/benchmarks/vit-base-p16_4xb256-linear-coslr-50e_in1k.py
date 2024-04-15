_base_ = [
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/default_runtime.py'
]

# checkpoint = "../diff-ijepa/logs/vitb16.224-8xb256-100e_in1k/jepa-ep100_.pth.tar"
# work_dir = "./work_dirs/in1k/ijepa/vitb16.224-8xb256-100e_in1k/linear"
checkpoint = "../diff-ijepa/logs/vitb16.224-8xb256-100e_in1k_diff-3x0cos-1e-4noisemse-uniformT/jepa-ep100_.pth.tar"
work_dir = "./work_dirs/in1k/ijepa/vitb16.224-8xb256-100e_in1k_diff-3x0cos-1e-4noisemse-uniformT/linear"

auto_scale_lr = dict(base_batch_size=16384, enable=True)

# dataset settings
# data_root = "/ssd/datasets/imagenet"
data_root = "/mnt/data/wangbei/datasets/imagenet"
train_dataloader = dict(batch_size=256, drop_last=True, dataset=dict(data_root=data_root,))
val_dataloader = dict(drop_last=False, dataset=dict(data_root=data_root,))
test_dataloader = dict(drop_last=False, dataset=dict(data_root=data_root,))

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
    )

# optimizer
optim_wrapper = dict(
    # _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='LARS', lr=0.05, weight_decay=0.0, momentum=0.9))

# learning rate scheduler
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[15, 30, 45], gamma=0.1)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=50)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=100))

randomness = dict(seed=0, diff_rank_seed=True)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=5),
    dist_cfg=dict(backend='nccl'))