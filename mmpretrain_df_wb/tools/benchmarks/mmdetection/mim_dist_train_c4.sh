#!/usr/bin/env bash

set -x

CFG=$1
PRETRAIN=$2  # pretrained model
GPUS=$3
PY_ARGS=${@:4}
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
export NCCL_P2P_LEVEL=NVL

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mim train mmdet $CFG \
    --launcher pytorch -G $GPUS \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$PRETRAIN \
    model.backbone.init_cfg.prefix="backbone." \
    model.roi_head.shared_head.init_cfg.type=Pretrained \
    model.roi_head.shared_head.init_cfg.checkpoint=$PRETRAIN \
    model.roi_head.shared_head.init_cfg.prefix="backbone." \
    $PY_ARGS
