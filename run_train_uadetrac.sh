#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python3 -m tools.train_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --train-dir ./demos/models/uadt_attention --model-suffix uadt_attention MODEL.DEVICE "cuda:0" MODEL.TRACK_HEAD.USE_ATTENTION "True"
