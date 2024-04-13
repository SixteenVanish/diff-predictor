_base_ = [
    '../_base_/schedules/imagenet_lars_coslr_200e.py',
    '../_base_/default_runtime.py',
]

# 参照ReSSL设置

work_dir = "./work_dirs/in1k/byol/byol_resnet50_4xb128-coslr-50e_in1k_mc1010_diff-mlp-sgd-4x0cos-1e-4noisemse-uniformT"

# optimizer
optimizer = dict(type='LARS', lr=4.8, momentum=0.9, weight_decay=1e-6)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        }),
)
auto_scale_lr = dict(base_batch_size=4096, enable=True)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
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
        type='CosineAnnealingLR', T_max=40, by_epoch=True, begin=10, end=50)
]

# model settings
diff_cfg = dict(
    model=dict(
        type='MLP', # MLP, Transformer
        in_channels=256,
        out_channels=256,
        mlp_params=dict(
            num_layers=2,
            mlp_dim=4096,
            bn_on=True,
            bn_sync_num=False,
            global_sync=False,),
        trans_params = dict(
            depth=6,
            predictor_embed_dim=384,
            num_attention_heads=12,
        ),
    ),
    scheduler=dict(
        num_train_timesteps=1000,
        prediction_type='epsilon'
    ),
    loss=dict(
        loss_noise_mse=1e-4,
        loss_noise_cos=0,
        loss_x0_cos=4,
        loss_x0_mse=0,
        loss_x0_smooth_l1=0,
        loss_inferx0_cos=0,
    ),
    distT_cfg = dict(
        type='uniform',
        ratio=[0.5, 1]
    ),
    cat_or_add='cat',
    pred_residual=False,
    remove_condition_T=False,
    test = None
)

model = dict(
    type='DiffBYOL',
    base_momentum=0.01,
    backbone=dict(
        type='ResNet',
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

    loss_byol_cos=0,
    decoder_cfg=None,
    # dict(
    #     online_img_size=0,
    #     target_img_size=0,
    #     decoder_layer='neck'
    # ),
    crop_cfg = dict(
        num_views=[2, 6],    # default:[2, 0]. If num_views is not [2, 0], then train_pipeline need to be changed.
        pred_map=[1, 0, 1, 0],  # [GpG, GpL, LpG, LpL]
    ),
    diff_pred=True,
    diff_cfg=diff_cfg,
    loss_add_cos=0,
)

# only keeps the latest 3 checkpoints
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3),
)

# data_root='/test2/datasets/imagenet'
data_root = '/mnt/data/wangbei/datasets/imagenet'
# data_root = '/home/gao2/disk/datasets/imagenet'
scale_m = 0.3
global_view_pipeline1 = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(scale_m, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=1.),
    dict(type='Solarize', thr=128, prob=0.),
    dict(type='RandomFlip', prob=0.5),
]
global_view_pipeline2 = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(scale_m, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.1),
    dict(type='Solarize', thr=128, prob=0.2),
    dict(type='RandomFlip', prob=0.5),
]
local_view_pipeline = [
    dict(
        type='RandomResizedCrop',
        scale=96,
        crop_ratio_range=(0.05, scale_m),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
    dict(type='Solarize', thr=128, prob=0.),
    dict(type='RandomFlip', prob=0.5),
]
num_crops = [1, 1, 6]   # default:[2,0] DINO:[1,1,6] SwAV:[2,6]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=num_crops, transforms=[global_view_pipeline1, global_view_pipeline2, local_view_pipeline]),
    dict(type='PackInputs')
]

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        split='train',
        pipeline=train_pipeline
        ))

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))