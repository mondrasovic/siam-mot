#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadetrac/DLA-34-FPN_box_EMM_UA_DETRAC_uadetrac/model_final.pth --test-dataset UA_DETRAC --set test --output-dir ./test_output_feature_nms --eval-csv-file eval_results_uadetrac.csv MODEL.TRACK_HEAD.USE_REID True MODEL.TRACK_HEAD.FREEZE_DORMANT False

