_base_ = [
    '../../_base_/datasets/cub_bs8_448.py',
    '../../_base_/schedules/cub_bs64.py',
    '../../_base_/default_runtime.py',
]

work_dir = './work_dirs/in100/byol/byol_resnet18_1xb512-coslr-100e_in100_trans/cub'
checkpoint = './work_dirs/in100/byol/byol_resnet18_1xb512-coslr-100e_in100_trans/epoch_100.pth'

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=200,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

data_root = '/mnt/data/wangbei/datasets/CUB_200_2011/CUB_200_2011'
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        data_root=data_root,
        split='train',
    ))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        data_root=data_root,
        split='test',
    ))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=70))

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
