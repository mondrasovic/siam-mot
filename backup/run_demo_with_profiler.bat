@echo off

python -m cProfile -o inference_profiler.profile -m demos.demo --demo-video E:/interreg_sample.mp4 --track-class person_vehicle --output-path E:/siammot_sandbox/tracking_vis