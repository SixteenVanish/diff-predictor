_base_ = 'mmdet::pascal_voc/faster-rcnn_r50-caffe-c4_ms-18k_voc0712.py'
# https://github.com/open-mmlab/mmdetection/blob/main/configs/pascal_voc/faster-rcnn_r50-caffe-c4_ms-18k_voc0712.py

data_preprocessor_default = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor_default,
    backbone=dict(
        depth=18,
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    rpn_head=dict(
        in_channels=256,
    ),
    roi_head=dict(
        shared_head=dict(
            type='ResLayerExtraNorm',
            depth=18,
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch'),
        bbox_head=dict(
            in_channels=512,
        )))

#train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=24000, val_interval=2000)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24000,
        by_epoch=False,
        milestones=[18000, 22000],
        gamma=0.1)
]

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000))

log_processor = dict(by_epoch=False)

custom_imports = dict(
    imports=['mmpretrain.models.utils.res_layer_extra_norm'],
    allow_failed_imports=False)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
data_root = "/mnt/data/wangbei/datasets/VOCdevkit"
train_dataloader = dict(
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='VOCDataset',
                data_root=data_root,
                ann_file='VOCtrainval_06-Nov-2007/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOCtrainval_06-Nov-2007/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args={{_base_.backend_args}}),
            dict(
                type='VOCDataset',
                data_root=data_root,
                ann_file='VOC2012/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2012/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args={{_base_.backend_args}})
        ]))
val_dataloader = dict(dataset=dict(
    pipeline=test_pipeline,
    data_root=data_root,
    ann_file='VOCtest_06-Nov-2007/ImageSets/Main/test.txt',
    data_prefix=dict(sub_data_root='VOCtest_06-Nov-2007/'),
    ))
test_dataloader = val_dataloader