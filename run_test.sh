#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadetrac_emb_2/DLA-34-FPN_box_EMM_UA_DETRAC_uadetrac_emb_2/model_0030000.pth --test-dataset UA_DETRAC --set test --output-dir ./test_output_orig_emb_2_30000 --eval-csv-file eval_results_uadetrac.csv MODEL.TRACK_HEAD.USE_REID False MODEL.TRACK_HEAD.FREEZE_DORMANT False

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadetrac_emb_2/DLA-34-FPN_box_EMM_UA_DETRAC_uadetrac_emb_2/model_0015000.pth --test-dataset UA_DETRAC --set test --output-dir ./test_output_orig_emb_2_15000 --eval-csv-file eval_results_uadetrac.csv MODEL.TRACK_HEAD.USE_REID False MODEL.TRACK_HEAD.FREEZE_DORMANT False
