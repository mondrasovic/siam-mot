from typing import List, Dict

import numpy as np

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import AnnoEntity

from siammot.utils.entity_utils import bbs_iou


def build_data_filter_fn(dataset_key: str, *args, **kwargs):
    """
    Get dataset specific filter function list, if there is any
    """
    filter_fn = None
    
    if dataset_key == 'CRP':
        filter_fn = CRPFilter(*args, **kwargs)
    elif dataset_key.startswith('MOT'):
        filter_fn = MOTFilter(*args, **kwargs)
    elif dataset_key == 'AOT':
        filter_fn = AOTFilter(*args, **kwargs)
    elif 'DETRAC' in dataset_key:
        filter_fn = UADETRACFilter(*args, **kwargs)
    
    return filter_fn


class BaseFilter:
    def __init__(self) -> None:
        pass
    
    def _filter(self, entity: AnnoEntity, ignored_gt_entities=None) -> bool:
        return False
    
    def filter(self, entity: AnnoEntity, ignored_gt_entities=None) -> bool:
        return self._filter(entity, ignored_gt_entities)
    
    def __call__(
        self, entities: List[AnnoEntity], ignored_entities=None, meta_data=None
    ):
        """
            Check each entity whether it is valid or should be filtered (
            ignored).
            :param entities: A list of entities (for a single frame) to be
            evaluated
            :param ignored_entities: A list of ignored entities or a binary
            mask indicating ignored regions
            :param meta_data: The meta data for the frame (or video)
            :return: A list of valid entities and a list of filtered (
            ignored) entities
            """
        valid_entities = []
        filtered_entities = []
        
        for entity in entities:
            if self._filter(entity, ignored_entities):
                filtered_entities.append(entity)
            else:
                valid_entities.append(entity)
        
        return valid_entities, filtered_entities


class CRPFilter(BaseFilter):
    """
        A class for filtering JTA dataset entities during evaluation
        A gt entity will be filtered (ignored) if its id is -1 (negative)
        A predicted entity will be filtered (ignored) if it is matched to a
        ignored ground truth entity
        """
    
    def __init__(self, iou_thresh=0.2, is_train=False, **kwargs) -> None:
        """
        :param iou_thresh: a predicted entity which overlaps with any ignored
        gt entity with at least
         iou_thresh would be filtered
        """
        self.iou_thresh = iou_thresh
    
    def _filter(self, entity: AnnoEntity, ignored_gt_entities=None) -> bool:
        if ignored_gt_entities is None:
            if entity.id < 0:
                return True
        else:
            for entity_ in ignored_gt_entities:
                if bbs_iou(entity, entity_) >= self.iou_thresh:
                    return True
        return False


class MOTFilter(BaseFilter):
    """
    A class for filtering MOT dataset entities
    A gt entity will be filtered (ignored) if its visibility ratio is very low
    A predicted entity will be filtered (ignored) if it is matched to a
    ignored ground truth entity
    """
    
    def __init__(
        self,
        visibility_thresh=0.1,
        iou_thresh=0.5,
        is_train=False,
        **kwargs,
    ) -> None:
        self.visibility_thresh = visibility_thresh
        self.iou_thresh = iou_thresh
        self.is_train = is_train
    
    def _filter(self, entity: AnnoEntity, ignored_gt_entities=None) -> bool:
        if ignored_gt_entities is None:
            if self.is_train:
                # any entity whose visibility is below the pre-defined
                # threshold should be filtered out
                # meanwhile, any entity whose class does not have label
                # needs to be filtered
                if entity.blob['visibility'] < self.visibility_thresh or \
                    not any(k in ('person', '2', '7') for k in entity.labels):
                    return True
            else:
                if 'person' not in entity.labels or int(entity.id) < 0:
                    return True
        else:
            for entity_ in ignored_gt_entities:
                if bbs_iou(entity, entity_) >= self.iou_thresh:
                    return True
            return False


class AOTFilter(BaseFilter):
    """
    A class for filtering AOT entities
    A gt entity will be filtered if it falls into one the following criterion
      1. tracking id is not Helicopter1 or Airplane1
      2. range distance is larger than 1200
    """
    
    def __init__(
        self,
        range_distance_thresh=1200,
        iou_thresh=0.2,
        is_train=False,
        **kwargs
    ) -> None:
        self.range_distance_thresh = range_distance_thresh
        self.iou_thresh = iou_thresh
        self.is_train = is_train
    
    def _filter(self, entity: AnnoEntity, ignored_gt_entities=None) -> bool:
        if ignored_gt_entities is None:
            range_distance_m = np.inf
            if 'range_distance_m' in entity.blob:
                range_distance_m = entity.blob['range_distance_m']
            
            labels = []
            if entity.labels is not None:
                labels = entity.labels
            
            if ('intruder' not in labels) or \
                (range_distance_m >= self.range_distance_thresh):
                return True
        else:
            for entity_ in ignored_gt_entities:
                if entity_.bbox is not None:
                    if bbs_iou(entity, entity_) >= self.iou_thresh:
                        return True
        return False


class UADETRACFilter(BaseFilter):
    def __init__(
        self,
        iou_ignored_entity_thresh: float = 0.5,
        ignored_region_overlap_thresh: float = 0.1,
        is_train: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.iou_ignored_entity_thresh: float = iou_ignored_entity_thresh
        self.ignored_region_overlap_thresh: float =\
            ignored_region_overlap_thresh
        self.is_train: bool = is_train

        dataset = kwargs.get('dataset', [])
        self.ignored_regions: Dict[str, np.ndarray] = {}
        for sample_name, data_sample in dataset:
            boxes = data_sample.metadata['ignored_regions']
            boxes = self.xywh_boxes_to_xyxy_np_array(boxes)
            self.ignored_regions[sample_name] = boxes
    
    def _filter(self, entity: AnnoEntity, ignored_gt_entities=None) -> bool:
        if entity.id < 0:
            return True
        
        if not self.is_train:
            ignored_boxes = self.ignored_regions[entity.blob['sample_name']]
            if len(ignored_boxes) == 0:
                return False
            
            box = self.xywh_boxes_to_xyxy_np_array(entity.bbox)
            area_ratios = self.intersection_over_area(box, ignored_boxes)
            if np.any(area_ratios >= self.ignored_region_overlap_thresh):
                return True

            # if ignored_gt_entities is not None:
            #     for entity_ in ignored_gt_entities:
            #         iou = bbs_iou(entity, entity_)
            #         if iou >= self.iou_ignored_entity_thresh:
            #             return True

        return False
    
    @staticmethod
    def xywh_boxes_to_xyxy_np_array(boxes):
        boxes = np.atleast_2d(np.asfarray(boxes))
        boxes[:, 2:] += boxes[:, :2]

        return boxes
    
    @staticmethod
    def intersection_over_area(
        box: np.ndarray,
        boxes: np.ndarray,
        eps: float = 1e-8
    ) -> np.ndarray:
        assert (box.ndim == 2) and (box.shape[0] == 1)
        assert (boxes.ndim == 2) and (boxes.shape[1] == 4)
        assert box[:, :2] <= box[:, 2:], "box must be in 'xyxy' format"
        assert boxes[:, :2] <= boxes[:, 2:], "boxes must be in 'xyxy' format"

        coords_tl = np.maximum(boxes[:, :2], box[..., :2])
        coords_br = np.minimum(boxes[:, 2:], box[..., 2:])
        
        wh = (coords_br - coords_tl).clip(min=0)
        intersect_areas = wh[:, 0] * wh[:, 1]
        box_area = np.prod(box[0, 2:] - box[0, :2])
        intersect_ratios = intersect_areas.astype(np.float) / (box_area + eps)

        return intersect_ratios
