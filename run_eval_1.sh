python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_emb_triplet_ga/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_emb_triplet_ga/model_0070000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_emb_ga/uadt/triplet/fNMS/0070000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_nms

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_emb_triplet_ga/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_emb_triplet_ga/model_0080000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_emb_ga/uadt/triplet/fNMS/0080000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_nms

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_emb_triplet_ga/DLA-34-FPN_box_EMM_UA_DETRAC_uadt_emb_triplet_ga/model_0060000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_emb_ga/uadt/triplet/fNMS/0060000 MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS triplet MODEL.TRACK_HEAD.SOLVER_TYPE feature_nms
