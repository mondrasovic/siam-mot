"""
Implements the Generalized R-CNN for SiamMOT
"""
from typing import List, Optional, Tuple

from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from torch import nn, Tensor
from yacs.config import CfgNode

from .backbone.backbone_ext import build_backbone
from .roi_heads import build_roi_heads


class SiamMOT(nn.Module):
    """
    Main class for R-CNN. Currently supports boxes and tracks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and
             computes detections / tracks from it.
    """
    
    def __init__(self, cfg: CfgNode) -> None:
        super(SiamMOT, self).__init__()
        
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        
        self.track_memory = None
    
    def flush_memory(
        self,
        cache: Optional[Tuple[Tensor, List[BoxList], List[BoxList]]] = None
    ) -> None:
        self.track_memory = cache
    
    def reset_siammot_status(self) -> None:
        self.flush_memory()
        self.roi_heads.reset_roi_status()
    
    def forward(
        self,
        images: Tensor,
        targets: None = None,
        given_detection: None = None
    ) -> List[BoxList]:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        if self.roi_heads:
            x, result, roi_losses = self.roi_heads(
                features,
                proposals,
                targets,
                self.track_memory,
                given_detection
            )
            if not self.training:
                self.flush_memory(cache=x)
        
        else:
            raise NotImplementedError
        
        if self.training:
            losses = {}
            losses.update(roi_losses)
            losses.update(proposal_losses)
            return result, losses
        
        return result


def build_siammot(cfg: CfgNode) -> SiamMOT:
    siammot = SiamMOT(cfg)
    return siammot
