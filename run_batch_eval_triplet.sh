#!/bin/bash

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0050000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-original_0050000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0040000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-original_0040000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0060000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-original_0060000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0050000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-feature_emb_0050000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_emb

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0040000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-feature_emb_0040000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_emb

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0060000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-feature_emb_0060000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_emb

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0070000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-feature_emb_0070000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_emb

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0080000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-feature_emb_0080000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_emb

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_final.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-feature_emb_final MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_emb

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0030000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-feature_emb_0030000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_emb

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0070000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-original_0070000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0080000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-original_0080000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_final.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-original_final MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE original

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_triplet/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_triplet/model_0030000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir ./eval/uadt_loss-triplet_slr-original_0030000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE original
