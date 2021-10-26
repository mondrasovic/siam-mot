from yacs.config import CfgNode as CN


_C = CN()

# ------------------------------------------------------------------------------
# Processing.
_C.PROC = CN()

# Number of child processes to spawn. If set to None, then the number of
# available CPUs is used.
_C.PROC.N_JOBS = None

# Number of consecutive frames in a single chunk.
_C.PROC.CHUNK_SIZE = 50

# Zero-based indices of start frames to achieve a shift of the sliding window
# when evaluatings chunks.
_C.PROC.START_FRAME_IDXS = (0, 25)

# ------------------------------------------------------------------------------
# Evaluation
_C.EVAL = CN()

# MOT metrics to report for each individual inference.
_C.EVAL.METRICS = [
    'num_frames',  # Total number of frames.
    'num_matches',  # Total number of matches.
    'num_switches',  # Total number of track switches.
    'num_false_positives',  # Total number of false positives (false alarms).
    'num_misses',  # Total number of misses (false negatives).
    'num_objects',  # Total number of unique object appearances over all frames (ground-truth).
    'num_predictions',  # Total number of unique prediction appearances over all frames.
    'num_fragmentations',  # Total number of switches from tracked to not tracked.
    'mostly_tracked',  # Number of objects tracked for at least 80 percent of lifespan.
    'partially_tracked',  # Number of objects tracked between 20 and 80 percent of lifespan.
    'precision',  # Number of detected objects over sum of detected and false positives.
    'recall',  # Number of detections over number of objects.
    'idf1',  # ID measures: global min-cost F1 score.
    'mota',  # Multiple object tracker accuracy.
    'motp',  # Multiple object tracker precision.
]

# Intersection over union threshold for bounding box distance.
_C.EVAL.IOU_THRESH = 0.5

# Threshold for the ratio of the bounding box area over some ignored region to
# consider it ignored.
_C.EVAL.IGNORE_AREA_RATIO_THRESH = 0.1

# ------------------------------------------------------------------------------
_C.EVAL.ALIAS = CN()

cfg = _C
