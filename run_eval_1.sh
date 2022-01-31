python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_dsa_2x/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0015000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_dsa_2x/uadt/with_dsa/0015000 MODEL.TRACK_HEAD.USE_ATTENTION True

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_dsa_2x/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0020000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_dsa_2x/uadt/with_dsa/0020000 MODEL.TRACK_HEAD.USE_ATTENTION True

python3 -m tools.test_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_dsa_2x/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0025000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_dsa_2x/uadt/with_dsa/0025000 MODEL.TRACK_HEAD.USE_ATTENTION True
