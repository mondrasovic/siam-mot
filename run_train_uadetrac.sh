#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python3 -m tools.train_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --train-dir ./demos/models/uadt_emb --model-suffix uadt MODEL.DEVICE "cuda:0" MODEL.TRAIN_EMB_FREEZE_REST "True" MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS "triplet"
