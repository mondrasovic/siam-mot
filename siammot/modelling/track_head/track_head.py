from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from torch import Tensor
from yacs.config import CfgNode

from siammot.modelling.track_head.EMM.target_sampler import EMMTargetSampler
from siammot.modelling.track_head.EMM.track_core import EMM
from siammot.modelling.track_head.track_utils import TrackPool, TrackUtils
from siammot.utils import registry


class TrackHead(torch.nn.Module):
    def __init__(
        self,
        tracker: EMM,
        tracker_sampler: EMMTargetSampler,
        track_utils: TrackUtils,
        track_pool: TrackPool
    ) -> None:
        super(TrackHead, self).__init__()
        
        self.tracker: EMM = tracker
        self.sampler: EMMTargetSampler = tracker_sampler
        
        self.track_utils: TrackUtils = track_utils
        self.track_pool: TrackPool = track_pool
    
    def forward(
        self,
        features: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        proposals: Optional[List[BoxList]] = None,
        targets: None = None,
        track_memory: Optional[
            Tuple[Tensor, List[BoxList], List[BoxList]]] = None
    ) -> Union[Tuple[Dict[Any, Any], None, Dict[Any, Any]], Tuple[
        Dict[Any, Any], List[BoxList], Dict[Any, Any]]]:
        if self.training:
            return self.forward_train(features, proposals, targets)
        else:
            return self.forward_inference(features, track_memory)
    
    def forward_train(self, features, proposals=None, targets=None):
        """
        Perform correlation on feature maps and regress the location of the
        object in other frame
        :param features: a list of feature maps from different intermediary
        layers of feature backbone
        :param proposals:
        :param targets:
        """
        
        with torch.no_grad():
            track_proposals, sr, track_targets = self.sampler(
                proposals, targets
            )
        
        return self.tracker(
            features, track_proposals, sr=sr, targets=track_targets
        )
    
    def forward_inference(
        self,
        features: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        track_memory: Optional[
            Tuple[Tensor, List[BoxList], List[BoxList]]] = None
    ) -> Union[Tuple[Dict[Any, Any], None, Dict[Any, Any]], Tuple[
        Dict[Any, Any], List[BoxList], Dict[Any, Any]]]:
        track_boxes = None
        if track_memory is None:
            self.track_pool.reset()
        else:
            (template_features, sr, template_boxes) = track_memory
            if template_features.numel() > 0:
                return self.tracker(
                    features, template_boxes, sr=sr,
                    template_features=template_features
                )
        return {}, track_boxes, {}
    
    def reset_track_pool(self) -> None:
        """
        Reset the track pool
        """
        self.track_pool.reset()
    
    def get_track_memory(
        self,
        features: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        tracks: List[BoxList]
    ) -> Tuple[Tensor, List[BoxList], List[BoxList]]:
        assert (len(tracks) == 1)
        active_tracks = self._get_track_targets(tracks[0])
        
        # no need for feature extraction of search region if
        # the tracker is tracktor, or no trackable instances
        if len(active_tracks) == 0:
            import copy
            template_features = torch.tensor([], device=features[0].device)
            sr = copy.deepcopy(active_tracks)
            sr.size = [active_tracks.size[0] + self.track_utils.pad_pixels * 2,
                active_tracks.size[1] + self.track_utils.pad_pixels * 2]
            track_memory = (template_features, [sr], [active_tracks])
        
        else:
            track_memory = self.tracker.extract_cache(features, active_tracks)
        
        track_memory = self._update_memory_with_dormant_track(track_memory)
        self.track_pool.update_cache(track_memory)
        
        return track_memory
    
    def _update_memory_with_dormant_track(
        self,
        track_memory: Tuple[Tensor, List[BoxList], List[BoxList]]
    ) -> Tuple[Tensor, List[BoxList], List[BoxList]]:
        cache = self.track_pool.get_cache()
        if not cache or track_memory is None:
            return track_memory
        
        dormant_caches = []
        for dormant_id in self.track_pool.get_dormant_ids():
            if dormant_id in cache:
                dormant_caches.append(cache[dormant_id])
        cached_features = [x[0][None, ...] for x in dormant_caches]
        if track_memory[0] is None:
            if track_memory[1][0] or track_memory[2][0]:
                raise Exception("Unexpected cache state")
            track_memory = [[]] * 3
            buffer_feat = []
        else:
            buffer_feat = [track_memory[0]]
        features = torch.cat(buffer_feat + cached_features)
        sr = cat_boxlist(track_memory[1] + [x[1] for x in dormant_caches])
        boxes = cat_boxlist(track_memory[2] + [x[2] for x in dormant_caches])
        return features, [sr], [boxes]
    
    def _get_track_targets(self, target: BoxList) -> BoxList:
        if len(target) == 0:
            return target
        active_ids = self.track_pool.get_active_ids()
        
        ids = target.get_field('ids').tolist()
        idxs = torch.zeros(
            (len(ids),), dtype=torch.bool, device=target.bbox.device
        )
        for _i, _id in enumerate(ids):
            if _id in active_ids:
                idxs[_i] = True
        
        return target[idxs]


def build_track_head(
    cfg: CfgNode,
    track_utils: TrackUtils,
    track_pool: TrackPool
) -> TrackHead:
    import siammot.modelling.track_head.EMM.track_core as track_core
    import siammot.modelling.track_head.EMM.target_sampler as target_sampler
    
    track_core  # To avoid the code clean-up routines to delete the import.
    target_sampler
    
    tracker = registry.SIAMESE_TRACKER[
        cfg.MODEL.TRACK_HEAD.MODEL
    ](cfg, track_utils)
    
    tracker_sampler = registry.TRACKER_SAMPLER[
        cfg.MODEL.TRACK_HEAD.MODEL
    ](cfg, track_utils)
    
    return TrackHead(tracker, tracker_sampler, track_utils, track_pool)
