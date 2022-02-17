python3 -m tools.test_net --local_rank 0 --config-file ./configs/dla/DLA_34_FPN_EMM_MOT17_fri.yaml --model-file ./demos/models/mot_dsa_2x/DLA-34-FPN_box_EMM_UA_MOT17_mot/model_0050000.pth --test-dataset MOT17 --set test --eval-csv-file eval_results.csv --output-dir eval_mot_dsa_2x/mot/with_dsa/0050000 MODEL.DEVICE "cuda:0" MODEL.TRACK_HEAD.ATTENTION.ENABLE True

python3 -m tools.test_net --local_rank 0 --config-file ./configs/dla/DLA_34_FPN_EMM_MOT17_fri.yaml --model-file ./demos/models/mot_dsa_2x/DLA-34-FPN_box_EMM_UA_MOT17_mot/model_0060000.pth --test-dataset MOT17 --set test --eval-csv-file eval_results.csv --output-dir eval_mot_dsa_2x/mot/with_dsa/0060000 MODEL.DEVICE "cuda:0" MODEL.TRACK_HEAD.ATTENTION.ENABLE True
