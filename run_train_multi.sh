#!/bin/bash

export CUDA_AVAILABLE_DEVICES=0,1
export WORLD_SIZE=2

python3 -m torch.distributed.launch --nproc_per_node=2 tools/train_net.py --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --train-dir ./demos/models/uadetrac --model-suffix uadetrac
