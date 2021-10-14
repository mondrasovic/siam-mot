#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadetrac/DLA_34_FPN_EMM_UADETRAC_uadetrac/model_final.pth --set train --output-dir ./test_output
