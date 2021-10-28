@echo off

setlocal EnableDelayedExpansion

set "visualizer_script=tools/solver_debug_visualizer.py"
set "data_subset=Train"
set "dataset_dir_path=E:\datasets\UA-DETRAC_GluonCV\raw_data\Insight-MVT_Annotation_%data_subset%"
set "out_dir_path=E:\solver_debug_visualization"

set test_dir_name=%1
set model_dir_name=DLA-34-FPN_box_EMM_UA_DETRAC

for %%i in (%test_dir_name%/%model_dir_name%/track_solver_debug_MVI_*.json) do (
    echo -----------------------------------------------------------------------
    set "base_name=%%~ni"
    set "sample_name=!base_name:~-9!"
    echo Processing sample !sample_name!
    set "curr_imgs_dir_path=%dataset_dir_path%\!sample_name!"
    set "curr_out_dir_path=%out_dir_path%\!sample_name!"
    
    if exist !curr_out_dir_path!\ (
        echo Skipping !sample_name!
    ) else (
        python %visualizer_script% !curr_imgs_dir_path! !curr_out_dir_path! %test_dir_name%/%model_dir_name%/%%i
        rem -i "input" -i "after NMS" -i "after ReID"
    )
)

endlocal
