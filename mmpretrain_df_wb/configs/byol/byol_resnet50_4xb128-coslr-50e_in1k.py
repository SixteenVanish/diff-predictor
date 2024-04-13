_base_ = [
    '../_base_/datasets/imagenet_bs32_byol.py',
    '../_base_/schedules/imagenet_lars_coslr_200e.py',
    '../_base_/default_runtime.py',
]

# 参照ReSSL设置

work_dir = "./work_dirs/in1k/byol/byol_resnet50_4xb128-coslr-50e_in1k"

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
# todo 其他参数设置
# model settings
diff_cfg = dict(
    model=dict(
        type='MLP', # MLP, Transformer
        
        num_layers=2,
        in_channels=256,
        out_channels=256,
        
        mlp_params=dict(
            mlp_dim=4096,
            bn_on=True,
            bn_sync_num=False,
            global_sync=False,),
        trans_params = dict(
            num_attention_heads=32,
            attention_head_dim=64,
        ),
    ),
    scheduler=dict(
        num_train_timesteps=1000,
        prediction_type='epsilon'
    ),
    loss=dict(
        loss_noise_mse=0,
        loss_noise_cos=0,
        loss_x0_cos=0,
        loss_x0_mse=0,
        
        loss_align_weight=0,
    ),
    diff_prob=1,
    cat_or_add='cat',
    conditioned_on_tgt=False,
    pred_residual=False,
    remove_condition_T=False,
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

    loss_byol_mse=0,
    loss_byol_cos=1,
    img_size=224,
    
    diff_pred=False,
    diff_cfg=diff_cfg,
    
    test_cfg = dict(
        distributionAnalysis = True,
        vis_tgt_cos = False,
        pic_path = work_dir,
        
        show_dfcos=False,
        show_dfcos_freq=50,
    ),
    
)

# only keeps the latest 3 checkpoints
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
)

# data_root='/test2/datasets/imagenet'
data_root = '/home/wangbei/datasets/imagenet'
# data_root = '/home/gao2/disk/datasets/imagenet'
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
        #pipeline=train_pipeline
        ))

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0))