from typing import List

import torch
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from torchvision.ops.boxes import nms
from yacs.config import CfgNode
from scipy.optimize import linear_sum_assignment

from siammot.modelling.track_head.track_utils import TrackPool
from siammot.modelling.reid.reid_man import ReIdManager, build_reid_manager


def boxlist_select_tensor_subset(boxlist: BoxList, mask) -> BoxList:
    box = boxlist.bbox[mask]
    scores = boxlist.get_field('scores')[mask]
    ids = boxlist.get_field('ids')[mask]

    boxlist = BoxList(box, boxlist.size, boxlist.mode)
    boxlist.add_field('scores', scores)
    
    boxlist.add_field('ids', ids)

    return boxlist


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
        sim_thresh = 0.7

        detections = detections[0]
        if len(detections) == 0:
            return [detections]
        
        ids_orig = detections.get_field('ids')
        dormant_ids = self.track_pool.get_dormant_ids()

        device = ids_orig.device
        _subset = boxlist_select_tensor_subset

        # Perform NMS only on detections that do not belong to dormant tracks.
        non_dormant_mask = torch.tensor(
            [int(x) not in dormant_ids for x in ids_orig],
            device=device
        )
        non_dormant_detections = _subset(detections, non_dormant_mask)
        nms_detections = boxlist_nms(non_dormant_detections, nms_thresh)

        ids = nms_detections.get_field('ids')

        # Extract still unassigned detections after the NMS.
        unassigned_mask = ids < 0
        unassigned_detections = _subset(nms_detections, unassigned_mask)

        # Extract dormant detections.
        dormant_mask = torch.tensor(
            [int(x) in dormant_ids for x in ids],
            device=device
        )
        dormant_detections = _subset(detections, dormant_mask)

        # We need frame indices to access previously processed frames.
        dormant_frame_idxs = []
        for dormant_id in dormant_detections.get_field('ids'):
            last_frame_idx = self.track_pool.get_last_active_frame_idx(
                dormant_id.item()
            )
            dormant_frame_idxs.append(last_frame_idx)
        
        # Compute similarity values based on the cosine similarity between the
        # embedding vectors obtained via the ReID model.
        cos_sim_matrix = self.reid_man.calc_cosine_sim_matrix(
            unassigned_detections,
            None,
            dormant_detections,
            dormant_frame_idxs
        )

        # Convert the similarity scores from <-1, 1> interval into the <0, 1>.
        cos_sim_matrix = cos_sim_matrix.cpu().numpy()
        cos_sim_matrix = 1 - (np.arcos(cos_sim_matrix) / np.pi)

        # Solve the "linear sum assignment problem" using optimization.
        row_idxs, col_idxs = linear_sum_assignment(-cos_sim_matrix)

        # Obtain exact indices of the unassigned and the dormant detections.
        unassigned_idxs = torch.where(unassigned_mask)[0]
        dormant_idxs = torch.where(dormant_mask)[0]

        # Initialize a mask for all the preserved detections. The detections
        # that were dormant but will end up assigned in this stage will have to
        # be removed from the list. Only the bounding box from the new
        # detection will be preserved.
        preserved_mask = torch.full_like(ids, True)

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
        
        detections_ = _subset(detections, preserved_mask)

        ids_ = detections_.get_field('ids')
        scores_ = detections_.get_field('scores')

        # Detections that haven't been assigned yet while their score is high
        # will be used to initialize new tracks.
        start_idxs = torch.where((ids_ < 0) & (scores_ >= self.start_thresh))[0]

        for idx in start_idxs:
            ids_[idx] = self.track_pool.start_track()

        # Inactive tracks are the ones which have their score below a threshold
        # and the ID has already been assigned.
        inactive_mask = (ids_ >= 0) & (scores_ < self.track_thresh)
        inactive_idxs = torch.where(inactive_mask)[0]

        for idx in inactive_idxs:
            self.track_pool.suspend_track(ids_[idx].item())
        ids_[inactive_idxs] = -1
        
        # Some IDs may have been removed by the NMS.
        nms_removed_ids = set(ids_orig.tolist()) - set(ids.tolist())
        for id_ in nms_removed_ids:
            self.track_pool.suspend_track(id_)

        self.track_pool.expire_tracks()
        self.track_pool.increment_frame()
        
        return [detections_]


def build_track_solver(cfg: CfgNode, track_pool: TrackPool) -> TrackSolver:
    reid_man = build_reid_manager(cfg)
    track_solver = TrackSolver(
        track_pool,
        reid_man,
        cfg.MODEL.TRACK_HEAD.TRACK_THRESH,
        cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH,
        cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH
    )

    return track_solver
