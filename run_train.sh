#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python3 -m tools.train_net.py --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --train-dir ./demos/models/uadetrac --model-suffix uadetrac
