_base_ = [
    '../_base_/datasets/imagenet_bs512_mocov3.py',
    '../_base_/default_runtime.py',
]

work_dir = "./work_dirs/in100/mocov3/mocov3_resnet18_1xb512-amp-coslr-100e_in100_diff-mlp-sgd-1loss-uniformT"

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
        loss_noise_mse=0,
        loss_noise_cos=0,
        loss_x0_mse=0,
        loss_x0_cos=0,
        loss_x0_smooth_l1=0,
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

temperature = 1.0
model = dict(
    type='DiffMoCoV3',
    base_momentum=0.01,  # 0.01 for 100e and 300e, 0.004 for 1000e
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=False),
    neck=dict(
        type='NonLinearNeck',
        in_channels=512,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=True),
    head=dict(
        type='MoCoV3Head',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=False,
            with_last_bn_affine=False,
            with_last_bias=False,
            with_avg_pool=False),
        loss=dict(type='CrossEntropyLoss', loss_weight=2 * temperature),
        temperature=temperature,
        adjust_temperature=False),
    
    loss_original=0,
    loss_diff=1,
    decoder_cfg=None,
    crop_cfg = dict(
        num_views=[2, 0],    # default:[2, 0]. If num_views is not [2, 0], then train_pipeline need to be changed.
        pred_map=[1, 0, 0, 0],  # [GpG, GpL, LpG, LpL]
    ),
    diff_pred=True,
    diff_cfg=diff_cfg,    
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
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3, interval=10))
# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096, enable=True)

# dataset settings
view_pipeline1 = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.2, 1.),
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
view_pipeline2 = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.2, 1.),
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
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiView',
        num_views=[1, 1],
        transforms=[view_pipeline1, view_pipeline2],
    ),
    dict(type='PackInputs')
]

data_root = '/mnt/data/wangbei/datasets/imagenet/'
ann_file = '/mnt/data/wangbei/datasets/imagenet/train.txt_100_withlabel.txt'
# data_root = '/test2/datasets/imagenet/'
# ann_file = '/test2/datasets/imagenet/train.txt_100_withlabel.txt'
train_dataloader = dict(
    batch_size=512,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file=ann_file,
        with_label=True,
        split='',
        pipeline=train_pipeline)
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
