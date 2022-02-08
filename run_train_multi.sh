#!/bin/bash

export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    tools/train_net.py --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --train-dir ./demos/models/uadt_dsa_2x --model-suffix uadt MODEL.TRACK_HEAD.ATTENTION.ENABLE "True" MODEL.TRACK_HEAD.ATTENTION.SAMPLING_STRATEGY "all"
