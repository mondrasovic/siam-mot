@echo off
python -m tools.train_net.py --config-file ./configs/dla/DLA_34_FPN_EMM_MOT17_custom.yaml --train-dir ./demos/models/mot17_custom --model-suffix mot17_custom