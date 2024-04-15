_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_lars_coslr_90e.py',
    '../../_base_/default_runtime.py',
]

work_dir = './work_dirs/in1k/byol/byol_resnet50_4xb128-coslr-50e_in1k_mc1010_diff-mlp-sgd-4x0cos-1e-4noisemse-uniformT/linear'
checkpoint = './work_dirs/in1k/byol/byol_resnet50_4xb128-coslr-50e_in1k_mc1010_diff-mlp-sgd-4x0cos-1e-4noisemse-uniformT/epoch_50.pth'

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096, enable=True)

# data_root='/test2/datasets/imagenet/'
data_root = '/mnt/data/wangbei/datasets/imagenet/'
# data_root = '/home/gao2/disk/datasets/imagenet'
train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        data_root=data_root,
    ))
val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        data_root=data_root))

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
            prefix='backbone.')))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=100))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)

