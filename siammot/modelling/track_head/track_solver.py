from typing import List

import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from yacs.config import CfgNode
from scipy.optimize import linear_sum_assignment

from siammot.modelling.track_head.track_utils import TrackPool
from siammot.modelling.reid.reid_man import (
    ReIdManager, build_or_get_existing_reid_manager
)


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
    
    def forward(self, detections: List[BoxList]):
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
        
        ids_orig = detections.get_field('ids')
        scores_orig = detections.get_field('scores')

        active_ids = self.track_pool.get_active_ids()
        dormant_ids = self.track_pool.get_dormant_ids()

        device = ids_orig.device
        _subset = boxlist_select_tensor_subset

        active_mask = torch.tensor(
            [int(x) in active_ids for x in ids_orig],
            device=device
        )
        scores_orig[active_mask] += 1

        # Perform NMS only on detections that do not belong to dormant tracks.
        non_dormant_mask = torch.tensor(
            [int(x) not in dormant_ids for x in ids_orig],
            device=device
        )
        non_dormant_detections = _subset(detections, non_dormant_mask)
        non_dormant_ids = non_dormant_detections.get_field('ids')
        nms_detections = boxlist_nms(non_dormant_detections, nms_thresh)
                
        ids = nms_detections.get_field('ids')
        scores = nms_detections.get_field('scores')

        scores[scores >= 2.] = scores[scores >= 2.] - 2
        scores[scores >= 1.] = scores[scores >= 1.] - 1

        # Some IDs may have been removed by the NMS.
        nms_removed_ids = set(non_dormant_ids.tolist()) - set(ids.tolist())
        
        print("**************************************************************")
        print(ids_orig.tolist())
        print(detections.get_field('scores'))
        print(ids.tolist())
        print(nms_detections.get_field('scores'))
        print(f"IDS to suspend: {nms_removed_ids}.")
        print(f"active IDs: {active_ids}.")
        print(f"dormant IDs: {dormant_ids}.")

        for id_ in nms_removed_ids:
            if id_ >= 0:
                self.track_pool.suspend_track(id_)

        # Extract still unassigned detections after the NMS.
        unassigned_mask = ids < 0
        unassigned_detections = _subset(nms_detections, unassigned_mask)

        # Extract dormant detections.
        dormant_mask = torch.tensor(
            [int(x) in dormant_ids for x in ids],
            device=device
        )
        dormant_detections = _subset(nms_detections, dormant_mask)

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

                dormant_id = ids[dormant_idx].item()
                ids[unassigned_idx] = dormant_id
                preserved_mask[dormant_idx] = False
                self.track_pool.resume_track(dormant_id)
        
        detections_ = _subset(nms_detections, preserved_mask)

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
        
        return [detections_]


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
