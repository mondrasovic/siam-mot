import abc

from typing import List

import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.layers import nms as _box_nms
from yacs.config import CfgNode

from siammot.modelling.track_head.track_utils import TrackPool
from siammot.debug.track_solver_debug import\
    build_or_get_existing_track_solver_debugger


def boxlist_nms_idxs_only(
    boxlist: BoxList,
    nms_thresh: float,
    score_field: str = 'scores'
) -> BoxList:
    """Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist (BoxList)
        nms_thresh (float)
        score_field (str)
    """
    boxlist = boxlist.convert('xyxy')
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)

    keep = _box_nms(boxes, scores, nms_thresh)

    return keep


def get_nms_boxes(detection: BoxList):
    detection = boxlist_nms(detection, nms_thresh=0.5)

    _ids = detection.get_field('ids')
    _scores = detection.get_field('scores')

    _scores[_scores >= 2.] = _scores[_scores >= 2.] - 2.
    _scores[_scores >= 1.] = _scores[_scores >= 1.] - 1.

    return detection, _ids, _scores


class TrackSolver(torch.nn.Module, abc.ABC):
    def __init__(
        self,
        track_pool: TrackPool,
        track_thresh: float = 0.3,
        start_track_thresh: float = 0.5,
        resume_track_thresh: float = 0.4,
        nms_thresh: float = 0.5,
        add_debug: bool = False
    ) -> None:
        super().__init__()

        self.track_pool: TrackPool = track_pool
        self.track_thresh: float = track_thresh
        self.start_thresh: float = start_track_thresh
        self.resume_track_thresh: float = resume_track_thresh

        self.nms_thresh: float = nms_thresh

        self.solver_debugger = (
            build_or_get_existing_track_solver_debugger() if add_debug else None
        )

    def _add_debug(self, stage, detections, metadata=None):
        if self.solver_debugger:
            self.solver_debugger.add_detections(
                stage,
                detections,
                self.track_pool.get_active_ids(),
                self.track_pool.get_dormant_ids(),
                metadata=metadata
            )

    def _debug_save_frame(self):
        if self.solver_debugger:
            self.solver_debugger.save_frame()


class TrackSolverOrig(TrackSolver):
    def __init__(
        self,
        track_pool: TrackPool,
        track_thresh: float = 0.3,
        start_track_thresh: float = 0.5,
        resume_track_thresh: float = 0.4,
        nms_thresh: float = 0.5,
        add_debug: bool = False
    ) -> None:
        super().__init__(
            track_pool, track_thresh, start_track_thresh, resume_track_thresh,
            nms_thresh, add_debug
        )

    def forward(self, detection: List[BoxList]):
        """
        The solver is to merge predictions from detection branch as well as
        from track branch.
        The goal is to assign an unique track id to bounding boxes that are
        deemed tracked
        :param detection: it includes three set of distinctive prediction:
        prediction propagated from active tracks, (2 >= score > 1, id >= 0),
        prediction propagated from dormant tracks, (2 >= score > 1, id >= 0),
        prediction from detection (1 > score > 0, id = -1).
        :return:
        """
        assert len(detection) == 1
        detection = detection[0]

        if len(detection) == 0:
            return [detection]

        self._add_debug('input', detection)

        track_pool = self.track_pool

        all_ids = detection.get_field('ids')
        all_scores = detection.get_field('scores')
        active_ids = track_pool.get_active_ids()
        dormant_ids = track_pool.get_dormant_ids()
        device = all_ids.device

        active_mask = torch.tensor(
            [int(x) in active_ids for x in all_ids], device=device
        )

        # differentiate active tracks from dormant tracks with scores
        # active tracks, (3 >= score > 2, id >= 0),
        # dormant tracks, (2 >= score > 1, id >= 0),
        # By doing this, dormant tracks will be merged to active tracks
        # during nms if they highly overlap
        all_scores[active_mask] += 1.

        nms_detection, _, _ = get_nms_boxes(detection)
        self._add_debug('after NMS', nms_detection)

        combined_detection = nms_detection
        _ids = combined_detection.get_field('ids')
        _scores = combined_detection.get_field('scores')

        # start track ids
        start_idxs = ((_ids < 0) & (_scores >= self.start_thresh)).nonzero()

        # inactive track ids
        inactive_idxs = ((_ids >= 0) & (_scores < self.track_thresh))
        nms_track_ids = set(_ids[_ids >= 0].tolist())
        all_track_ids = set(all_ids[all_ids >= 0].tolist())
        # active tracks that are removed by nms
        nms_removed_ids = all_track_ids - nms_track_ids
        inactive_ids = set(_ids[inactive_idxs].tolist()) | nms_removed_ids

        # resume dormant tracks, if needed
        dormant_mask = torch.tensor(
            [int(x) in dormant_ids for x in _ids], device=device
        )
        resume_ids = _ids[dormant_mask & (_scores >= self.resume_track_thresh)]
        for _id in resume_ids.tolist():
            track_pool.resume_track(_id)

        for _idx in start_idxs:
            _ids[_idx] = track_pool.start_track()

        active_ids = track_pool.get_active_ids()
        for _id in inactive_ids:
            if _id in active_ids:
                track_pool.suspend_track(_id)

        # make sure that the ids for inactive tracks in current frame are
        # meaningless (< 0)
        _ids[inactive_idxs] = -1

        self._add_debug('output', combined_detection)
        self._debug_save_frame()

        track_pool.expire_tracks()
        track_pool.increment_frame()

        return [combined_detection]


def build_track_solver(cfg: CfgNode, track_pool: TrackPool) -> TrackSolver:
    track_thresh = cfg.MODEL.TRACK_HEAD.TRACK_THRESH
    start_track_thresh = cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH
    resume_track_thresh = cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH
    nms_thresh = cfg.MODEL.TRACK_HEAD.NMS_THRESH
    add_debug = cfg.MODEL.TRACK_HEAD.ADD_DEBUG

    solver_type = cfg.MODEL.TRACK_HEAD.SOLVER_TYPE
    if solver_type == 'original':
        track_solver = TrackSolverOrig(
            track_pool, track_thresh, start_track_thresh, resume_track_thresh,
            nms_thresh, add_debug
        )
    else:
        raise ValueError('unrecognized solver type')

    return track_solver
