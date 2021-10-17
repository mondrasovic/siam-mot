@echo off

python -m tools.train_net --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC.yaml --train-dir ./demos/models/uadetrac --model-suffix uadetrac
