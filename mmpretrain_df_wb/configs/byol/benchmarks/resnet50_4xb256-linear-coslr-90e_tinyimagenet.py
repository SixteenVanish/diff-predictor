_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/schedules/imagenet_lars_coslr_90e.py',
    '../../_base_/default_runtime.py',
]

work_dir = './work_dirs/tiny/byol/byol_resnet50_1xb256-coslr-200e_tinyimagenet_decoder/linear'
checkpoint = './work_dirs/tiny/byol/byol_resnet50_1xb256-coslr-200e_tinyimagenet_decoder/epoch_200.pth'

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096, enable=True)

# data_root='/test2/datasets/imagenet/tiny-imagenet-200_format'
data_root = '/home/wangbei/datasets/imagenet/tiny-imagenet-200_format'
# data_root = '/home/gao2/disk/datasets/imagenet/tiny-imagenet-200_format'
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
    batch_size=256,
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
    batch_size=256,
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

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
            prefix='backbone.')),
    head=dict(
        num_classes=200,)
    )

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)
val_cfg = dict()
test_cfg = dict()
