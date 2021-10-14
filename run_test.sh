#!/bin/bash

python -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadetrac/DLA_34_FPN_EMM_UADETRAC_uadetrac/model_final.pth --set train --output_dir ./test_output
