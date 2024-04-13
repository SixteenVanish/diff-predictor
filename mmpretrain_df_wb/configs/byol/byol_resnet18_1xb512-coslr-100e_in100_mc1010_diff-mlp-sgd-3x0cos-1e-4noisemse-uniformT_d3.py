_base_ = [
    '../_base_/default_runtime.py',
]

# for decoder
work_dir = "./work_dirs/in100/byol/byol_resnet18_1xb512-coslr-100e_in100_mc1010_diff-mlp-sgd-3x0cos-1e-4noisemse-uniformT_d3"

# model settings
diff_cfg = dict(
    model=dict(
        type='MLP', # MLP, Transformer
        in_channels=256,
        out_channels=4096,
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
        loss_x0_mse=0,
        loss_x0_cos=3,
        loss_x0_smooth_l1=0,
        loss_inferx0_cos=0
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

# model settings
model = dict(
    type='DiffBYOL',
    base_momentum=0.01,
    backbone=dict(
        type='ResNet',  # ResNet_CIFAR
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
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=4096,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False),
        loss=dict(type='CosineSimilarityLoss')),
    
    loss_byol_cos=0,
    decoder_cfg=dict(
        img_size=224,
        decoder_layer='predictor',
        loss_decoder=1,
        
        map_size_list = [56, 112, 224],
        fc_size = 14,
        fc_channel = 512
    ),
    crop_cfg = dict(
        num_views=[2, 6],    # default:[2, 0]. If num_views is not [2, 0], then train_pipeline need to be changed.
        pred_map=[1, 0, 1, 0],  # [GpG, GpL, LpG, LpL]
    ),
    diff_pred=True,
    diff_cfg=diff_cfg,
    loss_add_cos=0
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='LARS', lr=9.6, weight_decay=1e-6, momentum=0.9),
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        }),
)

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
        T_max=90,
        by_epoch=True,
        begin=10,
        end=100,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
# only keeps the latest 3 checkpoints
default_hooks = dict(
    checkpoint=dict(max_keep_ckpts=3, interval=10),
    )
# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096, enable=True)

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

train_dataloader = dict(
    batch_size=512,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ImageNet',
        data_root='/mnt/data/wangbei/datasets/imagenet/',
        ann_file='/mnt/data/wangbei/datasets/imagenet/train.txt_100_withlabel.txt',
        pipeline=train_pipeline,
    )
)

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

