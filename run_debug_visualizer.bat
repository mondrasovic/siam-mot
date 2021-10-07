@echo off

set "visualizer_script=tools/solver_debug_visualizer.py"
set "data_subset=Test"
set "dataset_dir_path=E:\datasets\UA-DETRAC_GluonCV\raw_data\Insight-MVT_Annotation_%data_subset%"
set "out_dir_path=E:\solver_debug_visualization"

for %%i in (track_solver_debug_*.json) do (
    echo -----------------------------------------------------------------------
    echo Processing... %%i
    set "basename=%%~ni"
    set "curr_imgs_dir_path=%dataset_dir_path%\%basename:~-9%"
    set "curr_out_dir_path=%out_dir_path%\%basename%"
    python %visualizer_script% %curr_imgs_dir_path% %curr_out_dir_path% %%i
)