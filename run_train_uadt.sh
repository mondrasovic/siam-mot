#!/bin/bash

python3 -m tools.train_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --train-dir ./demos/models/uadt_dsa_ga --model-suffix uadt MODEL.TRACK_HEAD.ATTENTION.ENABLE "True"
