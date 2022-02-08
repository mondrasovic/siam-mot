python3 -m tools.test_net --local_rank 1 --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_dsa_2x/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0035000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_dsa_2x/uadt/with_dsa/0035000 MODEL.DEVICE "cuda:1" MODEL.TRACK_HEAD.ATTENTION.ENABLE True

python3 -m tools.test_net --local_rank 1 --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml --model-file ./demos/models/uadt_dsa_2x/DLA-34-FPN_box_EMM_UA_DETRAC_uadt/model_0045000.pth --test-dataset UA_DETRAC --set test --eval-csv-file eval_results.csv --output-dir eval_dsa_2x/uadt/with_dsa/0045000 MODEL.DEVICE "cuda:1" MODEL.TRACK_HEAD.ATTENTION.ENABLE True
