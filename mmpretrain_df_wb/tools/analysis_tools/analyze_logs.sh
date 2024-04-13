#!/usr/bin/env bash

CONFIG=$1
unset LD_LIBRARY_PATH

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python $(dirname "$0")/analyze_logs.py \
    $CONFIG \
    ${@:2}
