python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig_mini/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0025000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_orig_mini/uadt/without_dsa/0025000 MODEL.TRACK_HEAD.USE_ATTENTION False

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig_mini/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0030000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_orig_mini/uadt/without_dsa/0030000 MODEL.TRACK_HEAD.USE_ATTENTION False

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig_mini/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0035000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_orig_mini/uadt/without_dsa/0035000 MODEL.TRACK_HEAD.USE_ATTENTION False

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_orig_mini/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0040000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_orig_mini/uadt/without_dsa/0040000 MODEL.TRACK_HEAD.USE_ATTENTION False
