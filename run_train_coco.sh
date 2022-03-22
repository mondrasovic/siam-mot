#!/bin/bash

python3 -m tools.train_net --config-file ./configs/dla/DLA_34_FPN_EMM_COCO_fri.yaml --train-dir ./demos/models/coco_dsa_ga --model-suffix coco MODEL.TRACK_HEAD.ATTENTION.ENABLE "True" GRAD_ACCUM.ENABLE "True"
