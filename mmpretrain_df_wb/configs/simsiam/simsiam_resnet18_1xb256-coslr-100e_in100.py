_base_ = [
    '../_base_/default_runtime.py',
]

work_dir = "./work_dirs/in100/simsiam/simsiam_resnet18_1xb256-coslr-100e_in100"

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
        loss_x0_mse=0,
        loss_x0_cos=3,
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

model = dict(
    type='DiffSimSiam',
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=512,
        hid_channels=2048,
        out_channels=2048,
        num_layers=3,
        with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        loss=dict(type='CosineSimilarityLoss'),
        predictor=dict(
            type='NonLinearNeck',
            in_channels=2048,
            hid_channels=512,
            out_channels=2048,
            with_avg_pool=False,
            with_last_bn=False,
            with_last_bias=True)),
    
    loss_original=1,
    decoder_cfg=None,
    # dict(
    #     img_size=224,
    #     decoder_layer='neck'
    # ),
    crop_cfg = dict(
        num_views=[2, 0],    # default:[2, 0]. If num_views is not [2, 0], then train_pipeline need to be changed.
        pred_map=[1, 0, 0, 0],  # [GpG, GpL, LpG, LpL]
    ),
    diff_pred=False,
    diff_cfg=diff_cfg,
    loss_add_cos=0
)

# optimizer
# set base learning rate
lr = 0.05
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=lr, weight_decay=1e-4, momentum=0.9),
    paramwise_cfg=dict(custom_keys={'predictor': dict(fix_lr=True)}))

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

# additional hooks
custom_hooks = [
    dict(type='SimSiamHook', priority='HIGH', fix_pred_lr=True, lr=lr)
]

# dataset settings
dataset_type = 'ImageNet'
data_root = '/mnt/data/wangbei/datasets/imagenet/'
ann_file = '/mnt/data/wangbei/datasets/imagenet/train.txt_100_withlabel.txt'
# data_root = '/test2/datasets/imagenet/'
# ann_file = '/test2/datasets/imagenet/train.txt_100_withlabel.txt'

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# The difference between mocov2 and mocov1 is the transforms in the pipeline
view_pipeline = [
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
                saturation=0.4,
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
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline))

auto_scale_lr = dict(base_batch_size=256, enable=True)
