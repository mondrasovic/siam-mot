from typing import List, Optional

import torch
from maskrcnn_benchmark.modeling.roi_heads.box_head.loss import (
    FastRCNNLossComputation, make_roi_box_loss_evaluator,
)
from maskrcnn_benchmark.modeling.roi_heads.box_head \
    .roi_box_feature_extractors import \
    make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_predictors import \
    make_roi_box_predictor
from maskrcnn_benchmark.structures.bounding_box import BoxList
from torch import Tensor
from yacs.config import CfgNode

from .inference import make_roi_box_post_processor, PostProcessor


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """
    
    def __init__(self, cfg: CfgNode, in_channels: int) -> None:
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(
            cfg, in_channels
        )
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels
        )
        self.post_processor: PostProcessor = make_roi_box_post_processor(cfg)
        self.loss_evaluator: FastRCNNLossComputation = \
            make_roi_box_loss_evaluator(
                cfg
            )
    
    def forward(
        self,
        features: [Tensor],
        proposals: List[BoxList],
        targets: Optional[List[BoxList]] = None
    ):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are
                returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
        
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        
        if not self.training:
            result = self.post_processor(
                (class_logits, box_regression), proposals
            )
            return x, result, {}
        
        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg: CfgNode, in_channels: int) -> ROIBoxHead:
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough,
    just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
