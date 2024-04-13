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
        type='RandomGrayscale',
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
    dict(type='MultiView', num_views=1, transforms=[[]]),
    dict(type='PackInputs')
]

test_dataloader = dict(
    batch_size=50,      # tiny imagenet每个类别有500张训练图像，50张验证图像和50张测试图像
    num_workers=8,
    drop_last=False,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset',
        # data_root='/home/wangbei/datasets/imagenet/tiny-imagenet-200_format/TestOfTrainSet',
        data_root='/home/wangbei/datasets/imagenet/tiny-imagenet-200_format',
        # data_prefix='train_4class',
        # data_prefix='train',
        data_prefix='val',
        pipeline=train_pipeline,
    )
)
test_cfg = dict()

