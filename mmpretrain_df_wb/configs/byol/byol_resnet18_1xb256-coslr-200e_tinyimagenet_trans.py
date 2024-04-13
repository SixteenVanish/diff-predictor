_base_ = [
    '../_base_/default_runtime.py',
]

# 参照ReSSL设置

work_dir = "./work_dirs/tmp/byol_resnet18_1xb256-coslr-200e_tinyimagenet_trans"

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=5e-4))
auto_scale_lr = dict(base_batch_size=256, enable=True)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)
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
    type='DiffBYOL_v0',
    base_momentum=0.01,
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=False),
    neck=dict(
        type='NonLinearNeck',
        in_channels=512,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='Transformer', # 'LatentPredictHead' 'Transformer'
        predictor=dict(
            num_layers=2,
            in_channels=256,
            out_channels=256,
            num_attention_heads=32,
            attention_head_dim=64,
            
            cross_attention_dim=None,
            only_cross_attention=False,
            proj_in=True,
            norm_out=True,
            proj_to_clip_embeddings=True,
            
            cat_or_add='add',
            remove_condition_T=False,
            ),
        loss=dict(type='CosineSimilarityLoss')),
    
    loss_byol_cos=1,
    diff_pred=False,
)

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(interval=10, max_keep_ckpts=3))

view_pipeline = [
    dict(
        type='mmpretrain.RandomResizedCrop',
        scale=64,
        crop_ratio_range=(0.2, 1.),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='mmpretrain.ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='mmpretrain.RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        # channel_weights=(0.114, 0.587, 0.2989)
    ),
    dict(
        type='mmpretrain.GaussianBlur',
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
    batch_size=256,
    num_workers=13,
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



