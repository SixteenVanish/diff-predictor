scale_m = 0.3
global_view_pipeline1 = [
    dict(
        type='mmpretrain.RandomResizedCrop',
        scale=224,
        crop_ratio_range=(scale_m, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='mmpretrain.ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='mmpretrain.RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='mmpretrain.GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=1.),
    dict(type='mmpretrain.Solarize', thr=128, prob=0.),
    dict(type='RandomFlip', prob=0.5),
]
global_view_pipeline2 = [
    dict(
        type='mmpretrain.RandomResizedCrop',
        scale=224,
        crop_ratio_range=(scale_m, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='mmpretrain.ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='mmpretrain.RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='mmpretrain.GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.1),
    dict(type='mmpretrain.Solarize', thr=128, prob=0.2),
    dict(type='RandomFlip', prob=0.5),
]
local_view_pipeline = [
    dict(
        type='mmpretrain.RandomResizedCrop',
        scale=96,
        crop_ratio_range=(0.05, scale_m),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='mmpretrain.ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='mmpretrain.RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='mmpretrain.GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
    dict(type='mmpretrain.Solarize', thr=128, prob=0.),
    dict(type='RandomFlip', prob=0.5),
]
num_crops = [1, 1, 6]   # default:[2,0] DINO:[1,1,6] SwAV:[2,6]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=num_crops, transforms=[global_view_pipeline1, global_view_pipeline2, local_view_pipeline]),
    dict(type='PackInputs')
]

test_dataloader = dict(
    batch_size=50,
    num_workers=8,
    drop_last=False,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ImageNet',
        data_root='/mnt/data/wangbei/datasets/imagenet/',
        ann_file='/mnt/data/wangbei/datasets/imagenet/val_10class.txt',
        pipeline=train_pipeline,
    )
)
test_cfg = dict()

