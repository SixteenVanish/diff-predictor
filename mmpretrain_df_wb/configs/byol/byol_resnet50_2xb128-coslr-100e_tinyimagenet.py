_base_ = [
    '../_base_/default_runtime.py',
]

# 参照ReSSL设置

work_dir = "./work_dirs/byol_resnet50_2xb128-coslr-100e_tinyimagenet"

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=5e-4))
auto_scale_lr = dict(base_batch_size=256, enable=True)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
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
        type='CosineAnnealingLR', T_max=190, by_epoch=True, begin=10, end=200)
]

# model settings
model = dict(
    type='BYOL',
    base_momentum=0.01,
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=False),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False),
        loss=dict(type='CosineSimilarityLoss')),
)

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=3))

view_pipeline = [
    dict(
        type='RandomResizedCrop',
        scale=64,
        crop_ratio_range=(0.2, 1.),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        # channel_weights=(0.114, 0.587, 0.2989)
    ),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),  # or prob=1.0
    # dict(type='Solarize', thr=128, prob=0.),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset',
        data_root='/home/wangbei/datasets/imagenet/tiny-imagenet-200_format',
        data_prefix='train',
        pipeline=train_pipeline,
    )
)
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[255*0.4802, 255*0.4481, 255*0.3975],
    std=[255*0.2302, 255*0.2265, 255*0.2262],
    to_rgb=True)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=5),
    dist_cfg=dict(backend='nccl'))



