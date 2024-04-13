_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../../_base_/schedules/imagenet_lars_coslr_90e.py',
    '../../_base_/default_runtime.py',
]

work_dir = './output/tmp2'

checkpoint = './output/byol_memrecon_pretrain_scaleLR/epoch_50.pth'

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096, enable=True)

data_root = '/datasets/imagenet'
# data_root = '/home/gao2/disk/datasets/imagenet'
train_dataloader = dict(
    batch_size=128,
    num_workers=13,
    dataset=dict(
        data_root=data_root,
    ))
val_dataloader = dict(
    batch_size=128,
    num_workers=13,
    dataset=dict(
        data_root=data_root))

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
            prefix='backbone.')))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)

