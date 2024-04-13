_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_sgd_coslr_100e.py',
    '../../_base_/default_runtime.py',
]

work_dir = './output/tmp2'

# checkpoint = './output/byol_memrecon_pretrain_scaleLR/epoch_50.pth'

auto_scale_lr = dict(base_batch_size=1024, enable=False)     # todo 训练后，确认log中学习率是否正确

data_root = '/datasets/imagenet'
# data_root = '/home/gao2/disk/datasets/imagenet'
train_dataloader = dict(
    batch_size=128,
    num_workers=13,
    dataset=dict(
        data_root=data_root,
        ann_file='./datasplit/train_1percent.txt'
    ))
val_dataloader = dict(
    batch_size=128,
    num_workers=13,
    dataset=dict(
        data_root=data_root))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=optimizer,
#     paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=1.)})) todo 10%的时候修改，backbone不同学习率

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=60, by_epoch=True, begin=0, end=60)
]

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.'),
    )
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=10)
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60)

