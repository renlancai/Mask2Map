#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --resume-from work_dirs/M2M_nusc_r50_pp_2Phase_22n22ep/epoch_12.pth  --launcher pytorch ${@:3} --deterministic
