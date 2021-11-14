#!/bin/bash

export CUDA_AVAILABLE_DEVICES=1

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadetrac_emb/DLA-34-FPN_box_EMM_UA_DETRAC_uadetrac_emb/model_0105000.pth --test-dataset UA_DETRAC --set test --output-dir ./test_output_orig_emb_105000 --eval-csv-file eval_results_uadetrac.csv MODEL.TRACK_HEAD.USE_REID False MODEL.TRACK_HEAD.FREEZE_DORMANT False
