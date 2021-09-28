from typing import List

import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.layers import nms as _box_nms
from yacs.config import CfgNode
from scipy.optimize import linear_sum_assignment

from siammot.modelling.track_head.track_utils import TrackPool
from siammot.modelling.reid.reid_man import (
    ReIdManager, build_or_get_existing_reid_manager
)
from siammot.modelling.track_head.track_solver_debug import (
    TrackSolverDebugger, build_or_get_existing_track_solver_debugger
)

def boxlist_nms_mask_only(
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
    score = boxlist.get_field(score_field)

    keep = _box_nms(boxes, score, nms_thresh)

    return keep


def boxlist_select_tensor_subset(
    boxlist: BoxList,
    mask: torch.Tensor
) -> BoxList:
    box_subset = boxlist.bbox[mask]
    boxlist_subset = BoxList(box_subset, boxlist.size, boxlist.mode)

    for field in boxlist.fields():
        field_data_subset = boxlist.get_field(field)[mask]
        boxlist_subset.add_field(field, field_data_subset)
    
    return boxlist_subset


def get_nms_boxes(detection: BoxList):
    detection = boxlist_nms(detection, nms_thresh=0.5)
    
    _ids = detection.get_field('ids')
    _scores = detection.get_field('scores')
    
    # adjust the scores to the right range
    # _scores -= torch.floor(_scores) * (_ids >= 0) * (torch.floor(
    # _scores) != _scores)
    # _scores[_scores >= 1.] = 1.
    
    _scores[_scores >= 2.] = _scores[_scores >= 2.] - 2.
    _scores[_scores >= 1.] = _scores[_scores >= 1.] - 1.
    
    return detection, _ids, _scores


class TrackSolver(torch.nn.Module):
    def __init__(
        self,
        track_pool: TrackPool,
        reid_man: ReIdManager,
        track_thresh: float = 0.3,
        start_track_thresh: float = 0.5,
        resume_track_thresh: float = 0.4,
    ) -> None:
        super(TrackSolver, self).__init__()
        
        self.reid_man: ReIdManager = reid_man
        self.track_pool: TrackPool = track_pool
        self.track_thresh: float = track_thresh
        self.start_thresh: float = start_track_thresh
        self.resume_track_thresh: float = resume_track_thresh

        self.solver_debugger: TrackSolverDebugger =\
            build_or_get_existing_track_solver_debugger()
    
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
        self.solver_debugger.save_frame()

        track_pool.expire_tracks()
        track_pool.increment_frame()
        
        return [combined_detection]

    def forward_new(self, detections: List[BoxList]):
        """
        The solver is to merge predictions from the detection branch as well as
        from the track branch. The goal is to assign a unique track ID to
        bounding boxes that are deemed tracked.

        :param detection: it includes three set of distinctive prediction:
            prediction propagated from active tracks (2 >= score > 1, id >= 0),
            prediction propagated from dormant tracks (2 >= score > 1, id >= 0),
            prediction from detection (1 > score > 0, id = -1).
        :return:
        """
        
        # Process only one frame at a time.
        assert len(detections) == 1

        # TODO Make NMS and similarity threshold parametric via configuration.
        nms_thresh = 0.5
        sim_thresh = 0.4

        detections = detections[0]
        if len(detections) == 0:
            return [detections]
        
        self._add_debug('input', detections)

        ids_orig = detections.get_field('ids')
        dormant_ids = self.track_pool.get_dormant_ids()

        device = ids_orig.device
        _subset = boxlist_select_tensor_subset

        # Perform NMS only on detections that do not belong to dormant tracks.
        dormant_mask = torch.tensor(
            [int(x) in dormant_ids for x in ids_orig],
            device=device
        )
        non_dormant_detections = _subset(detections, ~dormant_mask)
        non_dormant_ids = non_dormant_detections.get_field('ids')
        nms_detections = boxlist_nms(non_dormant_detections, nms_thresh)
        
        self._add_debug('after NMS', nms_detections)

        ids = nms_detections.get_field('ids')
        scores = nms_detections.get_field('scores')

        scores[scores >= 1.0] -= 1.0

        # Some IDs may have been removed by the NMS.
        nms_removed_ids = set(non_dormant_ids.tolist()) - set(ids.tolist())

        for id_ in nms_removed_ids:
            if id_ >= 0:
                self.track_pool.suspend_track(id_)

        # Extract still unassigned detections after the NMS.
        unassigned_mask = ids < 0
        unassigned_detections = _subset(nms_detections, unassigned_mask)

        # Extract dormant detections.
        dormant_detections = _subset(detections, dormant_mask)

        # We need frame indices to access previously processed frames.
        dormant_frame_idxs = [
            self.track_pool.get_last_active_frame_idx(id_.item())
            for id_ in dormant_detections.get_field('ids')
        ]

        current_frame_idx = self.track_pool.frame_idx
        unassigned_frame_idxs = [current_frame_idx] * len(unassigned_detections)

        # Compute similarity values based on the cosine similarity between the
        # embedding vectors obtained via the ReID model.
        cos_sim_matrix = self.reid_man.calc_cosine_sim_matrix(
            unassigned_frame_idxs,
            unassigned_detections,
            dormant_frame_idxs,
            dormant_detections
        )

        # Solve the "linear sum assignment problem" using optimization.
        cos_sim_matrix = cos_sim_matrix.cpu().numpy()
        row_idxs, col_idxs = linear_sum_assignment(-cos_sim_matrix)

        # Obtain exact indices of the unassigned and the dormant detections.
        if len(row_idxs) > 0:
            unassigned_idxs, = torch.where(unassigned_mask)
            dormant_idxs, = torch.where(dormant_mask)
            print("*" * 100)
            print(f"There really is something.")

        # Initialize a mask for all the preserved detections. The detections
        # that were dormant but will end up assigned in this stage will have to
        # be removed from the list. Only the bounding box from the new
        # detection will be preserved.
        preserved_mask = torch.full_like(ids, True, dtype=torch.bool)

        # Assign IDs from the dormant detections to the unassigned detections if
        # their similarity is above the threshold.
        for row, col in zip(row_idxs, col_idxs):
            if cos_sim_matrix[row, col] >= sim_thresh:
                unassigned_idx = unassigned_idxs[row]
                dormant_idx = dormant_idxs[col]

                dormant_id = ids_orig[dormant_idx].item()
                ids[unassigned_idx] = dormant_id
                preserved_mask[dormant_idx] = False
                self.track_pool.resume_track(dormant_id)
        
        detections_ = _subset(nms_detections, preserved_mask)

        self._add_debug('after ReID', detections_)

        ids_ = detections_.get_field('ids')
        scores_ = detections_.get_field('scores')

        # Detections that haven't been assigned yet while their score is high
        # will be used to initialize new tracks.
        start_idxs, = torch.where((ids_ < 0) & (scores_ >= self.start_thresh))

        for idx in start_idxs:
            ids_[idx] = self.track_pool.start_track()

        # Inactive tracks are the ones which have their score below a threshold
        # and the ID has already been assigned.
        inactive_mask = (ids_ >= 0) & (scores_ < self.track_thresh)
        inactive_idxs, = torch.where(inactive_mask)

        for idx in inactive_idxs:
            self.track_pool.suspend_track(ids_[idx].item())
        ids_[inactive_idxs] = -1
        
        self.track_pool.expire_tracks()
        self.track_pool.increment_frame()

        self._add_debug('output', detections_)
        self.solver_debugger.save_frame()
        
        return [detections_]

    def _add_debug(self, stage, detections):
        self.solver_debugger.add_detections(
            stage, detections, self.track_pool.get_active_ids(),
            self.track_pool.get_dormant_ids()
    )


def build_track_solver(cfg: CfgNode, track_pool: TrackPool) -> TrackSolver:
    reid_man = build_or_get_existing_reid_manager(cfg)
    track_solver = TrackSolver(
        track_pool,
        reid_man,
        cfg.MODEL.TRACK_HEAD.TRACK_THRESH,
        cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH,
        cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH
    )

    return track_solver
