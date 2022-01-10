python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig_mini/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0045000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_orig_mini/uadt/without_dsa/0045000 MODEL.TRACK_HEAD.USE_ATTENTION False

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig_mini/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0050000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_orig_mini/uadt/without_dsa/0050000 MODEL.TRACK_HEAD.USE_ATTENTION False

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig_mini/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0055000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_orig_mini/uadt/without_dsa/0055000 MODEL.TRACK_HEAD.USE_ATTENTION False

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig_mini/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0060000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_orig_mini/uadt/without_dsa/0060000 MODEL.TRACK_HEAD.USE_ATTENTION False
