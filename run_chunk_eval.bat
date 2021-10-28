@echo off

set first_experiment_name=test_output
set second_experiment_name=test_output_reid_no_dorm_update
set sample_num=40775
set subset_name=Test

set first_file=./%first_experiment_name%/DLA-34-FPN_box_EMM_UA_DETRAC/MVI_%sample_num%.json
set second_file=./%second_experiment_name%/DLA-34-FPN_box_EMM_UA_DETRAC/MVI_%sample_num%.json
set gt_file=../../datasets/UA-DETRAC_GluonCV/raw_data/DETRAC_public/540p-%subset_name%/MVI_%sample_num%_v3.xml

python ./tools/eval_analyzer/main.py -o eval_diff_%sample_num%.csv %first_file% %second_file% %gt_file%
