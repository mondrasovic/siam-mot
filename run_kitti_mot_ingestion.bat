@echo off

set script_path=./siammot/data/ingestion/ingest_kitti_mot.py
set dataset_dir_path=E:\datasets\KITTI-MOT_GluonCV

python %script_path% %dataset_dir_path%
