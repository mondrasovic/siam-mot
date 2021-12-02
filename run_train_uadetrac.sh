#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python3 -m tools.train_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --train-dir ./demos/models/uadt_emb1024_tripletR_ga --model-suffix uadt_emb1024_tripletR_ga MODEL.DEVICE "cuda:0" MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS "triplet"
