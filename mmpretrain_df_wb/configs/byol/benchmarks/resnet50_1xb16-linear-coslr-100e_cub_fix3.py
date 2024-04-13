_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/cub_bs8_448.py',
    '../../_base_/schedules/cub_bs64.py',
    '../../_base_/default_runtime.py',
]

work_dir = './work_dirs/in1k/byol/byol_resnet50_4xb128-coslr-50e_in1k/cub_fix3'
checkpoint = './work_dirs/in1k/byol/byol_resnet50_4xb128-coslr-50e_in1k/epoch_50.pth'

data_root = '/home/wangbei/datasets/CUB_200_2011/CUB_200_2011'
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

model = dict(
    type='ImageClassifier',
    backbone=dict(
        frozen_stages=3,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    head=dict(num_classes=200, ))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=70))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))
