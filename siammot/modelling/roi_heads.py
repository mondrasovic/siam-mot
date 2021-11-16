from torch._C import device
from siammot.modelling import reid
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from torch import Tensor
from yacs.config import CfgNode

from siammot.modelling.box_head.box_head import ROIBoxHead
from siammot.modelling.track_head.track_head import TrackHead
from siammot.modelling.track_head.track_solver import TrackSolverReid
from .box_head.box_head import build_roi_box_head
from .track_head.track_head import build_track_head
from .track_head.track_solver import build_track_solver
from .track_head.track_utils import build_track_utils


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a
    single head.
    """
    
    def __init__(
        self,
        cfg: CfgNode,
        heads: List[Union[Tuple[str, ROIBoxHead], Tuple[str, TrackHead], Tuple[
            str, TrackSolverReid]]]
    ) -> None:
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()

        # TODO Some time in the future, remove this ugly solution.
        self.freeze_dormant: bool = cfg.MODEL.TRACK_HEAD.FREEZE_DORMANT

    def forward(
        self,
        features: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        proposals: List[BoxList],
        targets: None = None,
        track_memory: Optional[
            Tuple[Tensor, List[BoxList], List[BoxList]]] = None,
        given_detection: None = None
    ) -> Tuple[Tuple[Tensor, List[BoxList], List[BoxList]], List[BoxList], Dict[
        Any, Any]]:
        losses = {}
        
        if given_detection is None:
            x, detections, loss_box = self.box(features, proposals, targets)
        else:
            # adjust provided detection
            if len(given_detection[0]) > 0:
                x, detections, loss_box = self.box(
                    features, given_detection, targets
                )
            else:
                x = features
                detections = given_detection
                loss_box = {}
        losses.update(loss_box)
        
        if self.cfg.MODEL.TRACK_ON:
            if self.freeze_dormant:
                template_boxes = self.extract_dormant_template_boxes(
                    track_memory
                )

            _, tracks, loss_track = self.track(
                features, proposals, targets, track_memory
            )
            losses.update(loss_track)
            
            # solver is only needed during inference
            if not self.training:
                if tracks is not None:
                    # Refine bounding boxes (tracks) using the RPN head while
                    # exploiting already extracted features.
                    tracks = self._refine_tracks(features, tracks)
                    detections = [cat_boxlist(detections + tracks)]
                
                if self.freeze_dormant:
                    self.replace_dormant_template_boxes(
                        detections, template_boxes
                    )

                # TODO Exploit features through the parameter? Yeah....
                detections = self.solver(detections, features)
                
                # Get the current state for tracking.
                # Extract fresh feature ROIs for ongoing detections.
                x = self.track.get_track_memory(features, detections)
        
        return x, detections, losses
    
    def extract_dormant_template_boxes(self, track_memory):
        dormant_template_boxes = {}        
        if not track_memory:
            return dormant_template_boxes
        
        _, _, (template_boxes,) = track_memory        
        if len(template_boxes) > 0:
            dormant_ids = self.solver.track_pool.get_dormant_ids()
            ids = template_boxes.get_field('ids')
            boxes = template_boxes.bbox
            
            for id_, box in zip(ids, boxes):
                id_ = id_.item()
                if id_ in dormant_ids:
                    dormant_template_boxes[id_] = box.clone()

        return dormant_template_boxes

    def replace_dormant_template_boxes(self, detections, template_boxes):
        detections, = detections
        dormant_ids = self.solver.track_pool.get_dormant_ids()
        ids = detections.get_field('ids').tolist()
        boxes = detections.bbox

        for i, id_ in enumerate(ids):
            if id_ in dormant_ids:
                prev_template_box = template_boxes.get(id_)
                if prev_template_box is not None:
                    boxes[i] = prev_template_box
    
    def reset_roi_status(self) -> None:
        """
        Reset the status of ROI Heads
        """
        if self.cfg.MODEL.TRACK_ON:
            self.track.reset_track_pool()
    
    def _refine_tracks(
        self,
        features: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        tracks: List[BoxList]
    ) -> List[BoxList]:
        """
        Use box head to refine the bounding box location
        The final vis score is an average between appearance and matching score
        """
        if len(tracks[0]) == 0:
            return tracks[0]
        track_scores = tracks[0].get_field('scores') + 1.
        # track_boxes = tracks[0].bbox
        _, tracks, _ = self.box(features, tracks)
        det_scores = tracks[0].get_field('scores')
        det_boxes = tracks[0].bbox
        
        if self.cfg.MODEL.TRACK_HEAD.TRACKTOR:
            scores = det_scores
        else:
            scores = (det_scores + track_scores) / 2.
        boxes = det_boxes
        
        r_tracks = BoxList(
            boxes, image_size=tracks[0].size, mode=tracks[0].mode
        )
        r_tracks.add_field('scores', scores)
        r_tracks.add_field('ids', tracks[0].get_field('ids'))
        r_tracks.add_field('labels', tracks[0].get_field('labels'))
        
        return [r_tracks]


def build_roi_heads(cfg: CfgNode, in_channels: int) -> CombinedROIHeads:
    # individually create the heads, that will be combined together
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(('box', build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.TRACK_ON:
        track_utils, track_pool = build_track_utils(cfg)
        track_head = build_track_head(cfg, track_utils, track_pool)
        roi_heads.append(('track', track_head))
        # solver is a non-learnable layer that would only be used during
        # inference
        roi_heads.append(
            ('solver', build_track_solver(cfg, track_pool, track_head))
        )
    
    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)
    
    return roi_heads
