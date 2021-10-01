@echo off

set script_path=./siammot/data/ingestion/ingest_uadetract.py
set dataset_dir_path=E:\datasets\UA-DETRAC_GluonCV
python %script_path% %dataset_dir_path%