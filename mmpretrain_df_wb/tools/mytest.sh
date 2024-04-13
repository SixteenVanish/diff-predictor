#!/usr/bin/env bash

CONFIG=$1
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
export NCCL_P2P_LEVEL=NVL
unset LD_LIBRARY_PATH

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/mytest.py \
    $CONFIG \
    ${@:2}
