#!/bin/bash

python3 -m tools.train_net \
    --config-file ./configs/dla/DLA_34_FPN_EMM_UADETRAC_fri.yaml \
    --train-dir ./demos/models/uadt_256prop_orig \
    --model-suffix uadt \
    MODEL.TRACK_HEAD.ATTENTION.ENABLE "False" \
    MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE "256" \
    MODEL.TRACK_HEAD.PROPOSAL_PER_IMAGE "256" \
    SOLVER.VIDEO_CLIPS_PER_BATCH "8" \
    GRAD_ACCUM.ENABLE "False"
