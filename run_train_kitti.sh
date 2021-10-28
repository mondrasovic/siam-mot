#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python3 -m tools.train_net --config-file ./configs/dla/DLA_34_FPN_EMM_KITTI_fri.yaml --train-dir ./demos/models/kitti --model-suffix kitti