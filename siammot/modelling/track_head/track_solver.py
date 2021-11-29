import abc

from typing import List, Tuple

import torch
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, boxlist_iou
from maskrcnn_benchmark.layers import nms as _box_nms
from yacs.config import CfgNode
from scipy.optimize import linear_sum_assignment

from siammot.modelling.track_head.EMM.track_loss import features_to_emb
from siammot.modelling.track_head.track_head import TrackHead
from siammot.modelling.track_head.track_utils import TrackPool
from siammot.modelling.reid.reid_man import (
    ReIdManager, build_or_get_existing_reid_manager
)
from siammot.modelling.track_head.track_solver_debug import\
    build_or_get_existing_track_solver_debugger


def cos_sim_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Computes a 2D cosine similarity matrix between two sets of vectors.

    Args:
        a (torch.Tensor): First list of vectors of shape (N, D).
        b (torch.Tensor): Second list of vectors of shape (M, D).
        eps (float, optional): Small value to avoid division by zero.
                               Defaults to 1e-8.

    Returns:
        torch.Tensor: A 2D cosine similarity matrix of shape (N, M).
        The cell at i-th row and j-th column represents the cosine similarity
        between the a[i] and b[j] D-dimensional vectors.
    """
    a_norm = a.norm(dim=1)[..., None]
    b_norm = b.norm(dim=1)[..., None]

    a_ = a / torch.clamp(a_norm, min=eps)
    b_ = b / torch.clamp(b_norm, min=eps)

    sim_matrix = torch.mm(a_, b_.T)

    return sim_matrix

def boxlist_feature_nms(
    boxes: BoxList,
    embs: torch.Tensor,
    iou_thresh_1: float = 0.4,
    iou_thresh_2: float = 0.8,
    cos_sim_thresh: float = 0.6,
    score_field: str = 'scores'
) -> BoxList:
    assert len(boxes) == len(embs)
    assert embs.ndim == 2
    assert 0 < iou_thresh_1 < iou_thresh_2 < 1

    scores = boxes.get_field(score_field)
    boxes = boxes.convert('xyxy')
    iou_scores = boxlist_iou(boxes, boxes)
    cos_sim_scores = cos_sim_matrix(embs, embs)

    kept_indices = []

    for i in torch.argsort(scores, descending=True).tolist():
        for j in kept_indices:
            iou = iou_scores[i, j]
            if iou > iou_thresh_2:
                break
            elif iou > iou_thresh_1:
                cos_sim = cos_sim_scores[i, j]
                if cos_sim > cos_sim_thresh:
                    break
        else:
            kept_indices.append(i)
    
    return boxes[kept_indices]


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
                stage, detections, self.track_pool.get_active_ids(),
                self.track_pool.get_dormant_ids(), metadata=metadata
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
    
    def forward(self, detection: List[BoxList], features=None):
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


class TrackSolverReid(TrackSolver):
    def __init__(
        self,
        track_pool: TrackPool,
        reid_man: ReIdManager,
        track_thresh: float = 0.3,
        start_track_thresh: float = 0.5,
        resume_track_thresh: float = 0.4,
        nms_thresh: float = 0.5,
        sim_thresh: float = 0.4,
        add_debug: bool = False
    ) -> None:
        super().__init__(
            track_pool, track_thresh, start_track_thresh, resume_track_thresh,
            nms_thresh, add_debug
        )
        
        self.reid_man: ReIdManager = reid_man
        self.sim_thresh: float = sim_thresh

    def forward(self, detections: List[BoxList], features=None):
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
        assert len(detections) == 1

        detections = detections[0]
        if len(detections) == 0:
            return [detections]
        
        self._add_debug('input', detections)

        nms_detections = self._nms_non_dormant_detections(detections)
        nms_scores = nms_detections.get_field('scores')
        nms_scores[nms_scores >= 1.0] -= 1.0

        self._add_debug('after NMS', nms_detections)

        orig_ids = detections.get_field('ids')
        nms_ids = nms_detections.get_field('ids')
        nms_removed_ids = set(orig_ids.tolist()) - set(nms_ids.tolist())
        for curr_id in nms_removed_ids:
            if curr_id >= 0:
                self.track_pool.suspend_track(curr_id)

        dormant_mask = self._build_dormant_mask(nms_detections)
        dormant_detections = nms_detections[dormant_mask]
        unassigned_mask = (
            (nms_ids < 0) & (nms_scores >= self.resume_track_thresh)
        )
        unassigned_detections = nms_detections[unassigned_mask]

        sim_assignment_res = self._calc_linear_sum_assignment_rows_cols(
            unassigned_detections, dormant_detections
        )
        cos_sim_matrix, row_idxs, col_idxs = sim_assignment_res
        if len(row_idxs) > 0:
            unassigned_idxs, = torch.where(unassigned_mask)
            dormant_idxs, = torch.where(dormant_mask)
        reid_preserved_mask = torch.full_like(nms_ids, True, dtype=torch.bool)
        for row, col in zip(row_idxs, col_idxs):
            if cos_sim_matrix[row, col] >= self.sim_thresh:
                unassigned_idx = unassigned_idxs[row]
                dormant_idx = dormant_idxs[col]
                dormant_id = nms_ids[dormant_idx].item()
                nms_ids[unassigned_idx] = dormant_id
                reid_preserved_mask[dormant_idx] = False
                self.track_pool.resume_track(dormant_id)
        
        reid_detections = nms_detections[reid_preserved_mask]
        reid_ids = reid_detections.get_field('ids')
        reid_scores = reid_detections.get_field('scores')

        self._add_debug('after ReID', reid_detections)

        start_idxs, = torch.where(
            (reid_ids < 0) & (reid_scores >= self.start_thresh)
        )
        for idx in start_idxs:
            reid_ids[idx] = self.track_pool.start_track()
        
        active_ids = self.track_pool.get_active_ids()
        inactive_mask = (reid_ids >= 0) & (reid_scores < self.track_thresh)
        inactive_idxs, = torch.where(inactive_mask)
        for idx in inactive_idxs:
            curr_id = reid_ids[idx].item()
            if curr_id in active_ids:
                self.track_pool.suspend_track(curr_id)
        reid_ids[inactive_idxs] = -1 
        
        self.track_pool.expire_tracks()
        self.track_pool.increment_frame()

        self._add_debug('output', reid_detections)
        self._debug_save_frame()
        
        return [reid_detections]

    def _nms_non_dormant_detections(self, detections: BoxList) -> BoxList:
        dormant_mask = self._build_dormant_mask(detections)
        non_dormant_mask = ~dormant_mask
        non_dormant_idxs, = torch.where(non_dormant_mask)
        non_dormant_detections = detections[non_dormant_mask]
        nms_non_dormant_keep_idxs = boxlist_nms_idxs_only(
            non_dormant_detections, self.nms_thresh
        )        
        
        nms_non_dormant_mask = torch.full(
            (len(non_dormant_detections),), False, dtype=torch.bool,
            device=dormant_mask.device
        )
        nms_non_dormant_mask[nms_non_dormant_keep_idxs] = True
        nms_preserved_mask = dormant_mask.clone()
        nms_preserved_mask[non_dormant_idxs] = nms_non_dormant_mask
        nms_detections = detections[nms_preserved_mask]

        return nms_detections
    
    def _build_dormant_mask(self, detections: BoxList) -> torch.Tensor:
        ids = detections.get_field('ids')
        dormant_ids = self.track_pool.get_dormant_ids()
        mask = torch.tensor(
            [int(x) in dormant_ids for x in ids], device=ids.device
        )
        return mask
    
    def _calc_linear_sum_assignment_rows_cols(
        self,
        unassigned_detections: BoxList,
        dormant_detections: BoxList
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dormant_frame_idxs = [
            self.track_pool.get_last_active_frame_idx(id_.item())
            for id_ in dormant_detections.get_field('ids')
        ]
        current_frame_idx = self.track_pool.frame_idx
        unassigned_frame_idxs = [current_frame_idx] * len(unassigned_detections)

        cos_sim_matrix = self.reid_man.calc_cosine_sim_matrix(
            unassigned_frame_idxs, unassigned_detections,
            dormant_frame_idxs, dormant_detections
        )
        cos_sim_matrix = cos_sim_matrix.cpu().numpy()
        row_idxs, col_idxs = linear_sum_assignment(-cos_sim_matrix)

        return cos_sim_matrix, row_idxs, col_idxs
    

class TrackSolverFeatureNMS(TrackSolver):
    def __init__(
        self,
        track_pool: TrackPool,
        track_head: TrackHead,
        track_thresh: float = 0.3,
        start_track_thresh: float = 0.5,
        resume_track_thresh: float = 0.4,
        iou_thresh_1: float = 0.4,
        iou_thresh_2: float = 0.8,
        cos_sim_thresh: float = 0.6,
        add_debug: bool = False
    ) -> None:
        super().__init__(
            track_pool, track_thresh, start_track_thresh, resume_track_thresh,
            -1, add_debug
        )

        self.track_head: TrackHead = track_head
        self.iou_thresh_1: float = iou_thresh_1
        self.iou_thresh_2: float = iou_thresh_2
        self.cos_sim_thresh: float = cos_sim_thresh

    def forward(self, detections: List[BoxList], features):
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
        assert len(detections) == 1
        detections = detections[0]
        
        if len(detections) == 0:
            return [detections]
        
        track_pool = self.track_pool
        
        all_ids = detections.get_field('ids')
        all_scores = detections.get_field('scores')
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

        embs = self._get_detections_embeddings(detections, features)
        embs_list = embs.detach().cpu().numpy().tolist()
        self._add_debug('input', detections, metadata={'embeddings': embs_list})
        nms_detection = boxlist_feature_nms(
            detections, embs, self.iou_thresh_1, self.iou_thresh_2,
            self.cos_sim_thresh
        )
        
        _ids = nms_detection.get_field('ids')
        _scores = nms_detection.get_field('scores')
        
        _scores[_scores >= 2.] -= 2
        _scores[_scores >= 1.] -= 1

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
    
    def _get_detections_embeddings(
        self,
        detections: BoxList,
        features
    ) -> torch.Tensor:
        cache = self.track_pool.get_cache()

        ids = detections.get_field('ids')
        new_detections = detections[ids < 0]
        new_detections_template_features = (
            self.track_head.extract_template_features(
                features, new_detections
            )
        )
        
        template_features_combined = []
        new_detections_idx = 0
        for id_ in ids.tolist():
            if id_ >= 0:
                template_features = cache[id_][0]
            else:
                template_features = (
                    new_detections_template_features[new_detections_idx]
                )
                new_detections_idx += 1
            
            if template_features.ndim == 3:
                template_features = torch.unsqueeze(template_features, dim=0)
            template_features_combined.append(template_features)
        
        template_features_combined = torch.cat(
            template_features_combined, dim=0
        )
        embs = self.track_head.features_to_embeddings(
            template_features_combined
        ) 

        return embs


def build_track_solver(
    cfg: CfgNode,
    track_pool: TrackPool,
    track_head: TrackHead
) -> TrackSolver:
    track_thresh = cfg.MODEL.TRACK_HEAD.TRACK_THRESH
    start_track_thresh = cfg.MODEL.TRACK_HEAD.START_TRACK_THRESH
    resume_track_thresh = cfg.MODEL.TRACK_HEAD.RESUME_TRACK_THRESH
    nms_thresh = cfg.MODEL.TRACK_HEAD.NMS_THRESH
    add_debug = cfg.MODEL.TRACK_HEAD.ADD_DEBUG

    solver_type = cfg.MODEL.TRACK_HEAD.SOLVER_TYPE
    if solver_type == 'ext_reid':
        reid_man = build_or_get_existing_reid_manager(cfg)
        cos_sim_thresh = cfg.MODEL.TRACK_HEAD.COS_SIM_THRESH

        track_solver = TrackSolverReid(
            track_pool, reid_man, track_thresh, start_track_thresh,
            resume_track_thresh, nms_thresh, cos_sim_thresh, add_debug
        )
    elif solver_type == 'feature_nms':
        iou_thresh_1 = cfg.MODEL.TRACK_HEAD.IOU_THRESH_1
        iou_thresh_2 = cfg.MODEL.TRACK_HEAD.IOU_THRESH_2
        cos_sim_thresh = cfg.MODEL.TRACK_HEAD.COS_SIM_THRESH

        track_solver = TrackSolverFeatureNMS(
            track_pool, track_head, track_thresh, start_track_thresh, 
            resume_track_thresh, iou_thresh_1, iou_thresh_2, cos_sim_thresh, 
            add_debug
        )
    elif solver_type == 'original':
        track_solver = TrackSolverOrig(
            track_pool, track_thresh, start_track_thresh, resume_track_thresh,
            nms_thresh, add_debug
        )
    else:
        raise ValueError('unrecognized solver type')

    return track_solver
