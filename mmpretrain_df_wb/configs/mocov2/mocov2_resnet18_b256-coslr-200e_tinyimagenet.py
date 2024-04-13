_base_ = [
    '../../../_base_/default_runtime.py',
]

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
    type='MoCo',
    queue_len=16384,
    feat_dim=128,
    momentum=1-0.996,
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        norm_cfg=dict(type='BN'),
        zero_init_residual=False),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.2))

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=3))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)

# dataset settings
# The difference between mocov2 and mocov1 is the transforms in the pipeline
view_pipeline = [
    dict(
        type='RandomResizedCrop',
        scale=64,
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

#train_pipeline = [
#    dict(type='LoadImageFromFile'),
#    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
#    dict(type='PackInputs')
#]
train_pipeline = [
    dict(type='LoadImageFromFileDiffusion'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline], use_gen_img=True, prob_gen_img=1.0),
    dict(type='PackInputs', meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'gen_img_path', 'gen_img_scale', 'gen_chosen_view'))
]


train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ImageNetDiffusionScales',
        data_root='data/imagenet/tiny-imagenet-200_format',
        split='',
        ann_file='train_original.txt',
        gen_data_root='data/imagenet/tiny-imagenet-200_format/train_variations/',
        gen_data_file='data/imagenet/tiny-imagenet-200_format/train_variations_noise0.txt',
        with_label=False,
        pipeline=train_pipeline))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=5),
    dist_cfg=dict(backend='nccl'))
# cifar10
#data_preprocessor = dict(
#    type='SelfSupDataPreprocessor',
#    mean=[255*0.4914, 255*0.4822, 255*0.4465],
#    std=[255*0.2023, 255*0.1994, 255*0.2010],
#    to_rgb=True)
# cifar100
#data_preprocessor = dict(
#    type='SelfSupDataPreprocessor',
#    mean=[255*0.5071, 255*0.4867, 255*0.4408],
#    std=[255*0.2675, 255*0.2565, 255*0.2761],
#    to_rgb=True)
# tiny imagenet
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[255*0.4802, 255*0.4481, 255*0.3975],
    std=[255*0.2302, 255*0.2265, 255*0.2262],
    to_rgb=True)
