#!/bin/bash

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_orig/model_0050000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-none_slr-original_0050000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS none MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_orig/model_0080000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-none_slr-original_0080000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS none MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_orig/model_final.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-none_slr-original_final MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS none MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_contrastive/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_contrastive/model_0060000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-contrastive_slr-original_0060000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS contrastive MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_contrastive/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_contrastive/model_0070000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-contrastive_slr-original_0070000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS contrastive MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_contrastive/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_contrastive/model_0080000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-contrastive_slr-original_0080000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS contrastive MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_contrastive/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_contrastive/model_0030000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-contrastive_slr-original_0030000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS contrastive MODEL.TRACK_HEAD.SOLVER_TYPE original
