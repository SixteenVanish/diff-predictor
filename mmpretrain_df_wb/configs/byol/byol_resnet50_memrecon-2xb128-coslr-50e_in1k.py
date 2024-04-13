_base_ = [
    '../_base_/datasets/imagenet_bs32_byol.py',
    '../_base_/schedules/imagenet_lars_coslr_200e.py',
    '../_base_/default_runtime.py',
]


work_dir = './output/tmp2'
# work_dir = './output/vis_byol_memrecon_pretrain_scaleLR'

load_from = './output/byol_memrecon_pretrain_scaleLR/epoch_50.pth'
resume = True

auto_scale_lr = dict(base_batch_size=4096, enable=True)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=51)
train_dataloader = dict(
    batch_size=128,
    num_workers=13,
    dataset=dict(
        data_root='/home/wangbei/datasets/imagenet'
        # data_root='/datasets/imagenet'
        # data_root = '/home/gao2/disk/datasets/imagenet'
    )
)

# test
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    isTest=False
)
test_dataloader = dict(
    batch_size=128,
    num_workers=13,
    dataset=dict(
        type='ImageNet',
        data_root='/datasets/imagenet/val'))
test_evaluator = dict(type='NoCalculate')
# test_evaluator = dict(type='Accuracy', topk=(1, 5))
test_cfg = dict()

cfg = dict(
    NUM_GPUS=1,
    MODEL=dict(
        ARCH="RESNET",
    ),
    RESNET=dict(
        WIDTH_PER_GROUP=64,
        DEPTH=50,
    ),
    MEMORY=dict(
        DIM_SOURCE_MEM=256,
        DIM_TARGET_MEM=256,
        DIM_CROSS_MEM=256,
        DIM_CROSS_INPUT_PROJ=256,

        RADIUS=16,
        N_SLOT=4096,
        N_HEAD=1,

        N_CONCEPT=4096,

        RECON_LOSS_TYPE='l2',
        AVERAGE_DIM=-1,
        USE_INPUT_PROJ=True,
        USE_OUTPUT_PROJ=False,
        NUM_MLP_LAYERS=2,
        NUM_MLP_LAYERS_IN=2,
        NUM_MLP_LAYERS_OUT=1,

        USE_SRC_RECON=False,
        USE_TAR_RECON=False,
        USE_ALIGN=False,
        KL_T=-2.,

        USE_SOURCE_MEM_CONTRASTIVE=False,
        USE_TARGET_MEM_CONTRASTIVE=False,
        USE_CROSS_MEM_CONTRASTIVE=False,

        ADDRESS_TYPE="dot",
        PREDICT_RESIDUAL=False,

        USE_GAUSSIAN_MEMORY=False,
        USE_GAUSSIAN_KL=False,
        GAUSSIAN_NORMAL=False,

        USE_SPARSE=False,
        USE_SPARSE_BEFORE=False,
        SPARSE_LOSS_TYPE='topk',
        SPARSE_TOPK=100,

        # loss weight
        LOSS_TARGET_MEM_CONTRASTIVE=0.,
        LOSS_SOURCE_MEM_CONTRASTIVE=0.,

        LOSS_SOURCE_RECON=0.1,
        LOSS_TARGET_RECON=0.1,

        LOSS_SOURCE_MEM_SPARSE=0.,
        LOSS_TARGET_MEM_SPARSE=0.,

        LOSS_PREDICT=1.,
        LOSS_KL=0.,
        LOSS_GAUSSIAN_KL=0.,
        NORM_MEM=False,
        NORM_INPUT=False,
        RECON_TYPE='mlp_v1',

        # Type of the structure of the predictor, memory for the key-value memory enhanced predictor and mlp for the mlp predictor
        CROSS_BRANCH_TYPE='memory',
        DINO_TEMP_T=0.05,  # Temperature for target in DINO loss
        DINO_TEMP_S=0.05,  # Temperature for source in DINO loss
    ),
    CONTRASTIVE=dict(
        T=0.5,  # default 0.07
        DIM=256,    # 128 default, if changed, change nCls too
        NUM_MLP_LAYERS=2,   # default 1
        BN_MLP=True,
        BN_SYNC_MLP=True,
        MLP_DIM=4096,
        SEQUENTIAL=True,    # def fault
        MOMENTUM=0.996,     # default 0.5
        MOMENTUM_ANNEALING=True,    # default false
        PREDICTOR_DEPTHS=[1],
        PREDICTOR_TYPE='MemoryRecon',
    ),
    BN=dict(
        USE_PRECISE_STATS=False,
        NUM_BATCHES_PRECISE=400,
        WEIGHT_DECAY=0.0,
        NUM_SYNC_DEVICES=1,
        NORM_TYPE="sync_batchnorm",
        GLOBAL_SYNC=False,
        # NORM_TYPE="sync_batchnorm_apex",
    ),
    TEST=dict(
        VIS_MIDDLE=True,
        VIS_PATH=work_dir,
    )
)

# model settings
model = dict(
    type='MemReconBYOL',
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
        type='MemReconHead',
        dim_input=256,
        dim_output=256,
        cfg=cfg,
    )
)

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

# runtime settings
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=45, by_epoch=True, begin=5, end=50)
]



