#!/usr/bin/env bash

IMG=$1
CONFIG=$2
CHECKPOINT=$3

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python $(dirname "$0")/vis_cam.py \
    $IMG \
    $CONFIG \
    $CHECKPOINT \
    ${@:4}
