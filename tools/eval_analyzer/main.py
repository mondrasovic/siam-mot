import argparse
import dataclasses
import itertools
import json
import sys

from xml.etree import ElementTree
from typing import Iterator, Tuple, Dict, Sequence, List, Union, Iterable

import motmetrics as mm
import numpy as np
import pandas as pd

from tabulate import tabulate

from config import cfg


def intersection_over_area(
    box: np.ndarray,
    boxes: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    assert (box.ndim == 2) and (box.shape[0] == 1)
    assert (boxes.ndim == 2) and (boxes.shape[1] == 4)

    coords_tl = np.maximum(boxes[:, :2], box[..., :2])
    coords_br = np.minimum(boxes[:, 2:], box[..., 2:])
    
    wh = (coords_br - coords_tl).clip(min=0)
    intersect_areas = wh[:, 0] * wh[:, 1]
    box_area = np.prod(box[0, 2:] - box[0, :2])
    intersect_ratios = intersect_areas.astype(np.float) / (box_area + eps)

    return intersect_ratios


BoxT = Tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class Entity:
    frame_idx: int
    obj_id: int
    box: BoxT


class TrackSampleMan:
    def __init__(
        self,
        sample_xml_file_path: str,
        ignore_area_ratio_thresh: float = 0.1
    ) -> None:
        assert 0 < ignore_area_ratio_thresh < 1

        self._ignored_regions: Union[List, np.ndarray] = []
        self._gt_entities_map: Dict[int, Sequence[Entity]]= {}
        self.ignore_area_ratio_thresh: float = ignore_area_ratio_thresh

        self._init_from_xml(sample_xml_file_path)

        self._ignored_regions = self.xywh_boxes_to_xyxy_np_array(
            self._ignored_regions
        )
    
    def __getitem__(self, frame_idx: int) -> Sequence[Entity]:
        return self._gt_entities_map[frame_idx]
    
    def is_box_valid(self, box: BoxT) -> bool:
        if len(self._ignored_regions) == 0:
            return True
        
        box = self.xywh_boxes_to_xyxy_np_array(box)
        area_ratios = intersection_over_area(box, self._ignored_regions)
        
        return np.all(area_ratios < self.ignore_area_ratio_thresh)
    
    def _init_from_xml(self, sample_xml_file_path: str) -> None:
        tree = ElementTree.parse(sample_xml_file_path)
        root = tree.getroot()

        for box in root.findall('./ignored_region/box'):
            box = self._read_box(box.attrib)
            self._ignored_regions.append(box)
        
        for frame in root.findall('./frame'):
            frame_num = int(frame.attrib['num'])
            frame_idx = frame_num - 1

            entities = []
            for target in frame.findall('.//target'):
                obj_id = int(target.attrib['id'])

                box_attr = target.find('box').attrib
                box = self._read_box(box_attr)

                entity = Entity(frame_idx, obj_id, box)
                entities.append(entity)
            
            self._gt_entities_map[frame_idx] = entities
 
    @staticmethod
    def _read_box(node_attr) -> BoxT:
        x = float(node_attr['left'])
        y = float(node_attr['top'])
        w = float(node_attr['width'])
        h = float(node_attr['height'])

        return (x, y, w, h)
    
    @staticmethod
    def xywh_boxes_to_xyxy_np_array(boxes) -> np.ndarray:
        boxes = np.atleast_2d(np.asfarray(boxes))
        boxes[:, 2:] += boxes[:, :2]

        return boxes
   

def parse_args():
    parser = argparse.ArgumentParser(
        description="SiamMOT evaluation inference analyzer.",
    )
    parser.add_argument(
        '-c', '--config', help="path to the YAML configuration file"
    )
    parser.add_argument(
        '-o', '--output', metavar='OUTFILE', help="output CSV file path"
    )
    parser.add_argument(
        'eval_file_path_1', metavar='FILE1',
        help="first evaluation inference JSON file path"
    )
    parser.add_argument(
        'eval_file_path_2', metavar='FILE2',
        help="second evaluation inference JSON file path"
    )
    parser.add_argument(
        'gt_file_path', metavar='GTFILE', help="ground-truth XML file path"
    )
    parser.add_argument(
        'opts', metavar='OPTIONS', nargs=argparse.REMAINDER,
        help="overwriting the default YAML configuration"
    )
    args = parser.parse_args()

    return args


def make_entity_from_json(data: Dict) -> Entity:
    frame_idx = data['blob']['frame_idx']
    obj_id = data['id']
    box = tuple(data['bb'])
    entity = Entity(frame_idx, obj_id, box)

    return entity


def iter_frames_chunks(
    entities: Iterable[Entity],
    chunk_size: int,
    start_frame_idx: int = 0
):
    assert chunk_size > 0, "chunk size must be positive"
    assert start_frame_idx >= 0, "start frame index must be non-negative"

    entities_filtered = filter(
        lambda e: e.frame_idx >= start_frame_idx, entities
    )
    for _, grouper in itertools.groupby(
        entities_filtered,
        key=lambda e: (e.frame_idx - start_frame_idx) // chunk_size
    ):
        yield grouper


def accum_frames_chunk_events(
    chunk_entities_iter: Iterator[Entity],
    track_sample_man: TrackSampleMan,
    iou_thresh: float = 0.5
) -> mm.MOTAccumulator:
    assert 0 < iou_thresh < 1, "IoU threshold must be within (0, 1) interval"

    def get_valid_entity_ids_and_boxes(
        entities: Iterable[Entity]
    ) -> Tuple[List[int], List[BoxT]]:
        ids, boxes = [], []

        for entity in entities:
            obj_id, box = entity.obj_id, entity.box

            if (obj_id >= 0) and track_sample_man.is_box_valid(box):
                ids.append(obj_id)
                boxes.append(box)
        
        return ids, boxes
    
    accumulator = mm.MOTAccumulator(auto_id=True)
    min_frame_idx, max_frame_idx = sys.maxsize, 0

    for frame_idx, out_entities_iter in itertools.groupby(
        chunk_entities_iter, key=lambda e: e.frame_idx
    ):
        gt_entities = track_sample_man[frame_idx]
        
        gt_ids, gt_boxes = get_valid_entity_ids_and_boxes(gt_entities)
        out_ids, out_boxes = get_valid_entity_ids_and_boxes(out_entities_iter)
        
        box_distances = mm.distances.iou_matrix(
            gt_boxes, out_boxes, max_iou=1 - iou_thresh
        )
        accumulator.update(gt_ids, out_ids, box_distances)

        min_frame_idx = min(min_frame_idx, frame_idx)
        max_frame_idx = max(max_frame_idx, frame_idx)
    
    if max_frame_idx == sys.maxsize:
        return False, None

    name = f'frames_{min_frame_idx:04d}_{max_frame_idx:04d}'

    return True, (name, accumulator)


def read_inference_entities(json_file_path: str) -> Sequence[Entity]:
    with open(json_file_path, 'rt') as in_file:
        content = json.load(in_file)
        entities = tuple(map(make_entity_from_json, content['entities']))

        return entities


def compute_frame_chunks_metrics_summary(
    entities: Iterable[Entity],
    track_sample_man: TrackSampleMan,
) -> pd.DataFrame:
    chunk_size = cfg.PROC.CHUNK_SIZE
    iou_thresh = cfg.EVAL.IOU_THRESH

    chunk_names, accumulators = [], []
    for start_frame_idx in cfg.PROC.START_FRAME_IDXS:
        chunks_iter = iter_frames_chunks(
            entities, chunk_size, start_frame_idx
        )
        for curr_chunk in chunks_iter:
            status, ret = accum_frames_chunk_events(
                curr_chunk, track_sample_man, iou_thresh
            )
            if status:
                name, accumulator = ret
                chunk_names.append(name)
                accumulators.append(accumulator)
    
    metrics_host = mm.metrics.create()
    summary = metrics_host.compute_many(
        accumulators, metrics=cfg.EVAL.METRICS, names=chunk_names,
        generate_overall=False
    )

    return summary


def main() -> int:
    args = parse_args()

    if args.config:
        cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    track_sample_man = TrackSampleMan(
        args.gt_file_path, cfg.EVAL.IGNORE_AREA_RATIO_THRESH
    )

    entities_1 = read_inference_entities(args.eval_file_path_1)
    entities_2 = read_inference_entities(args.eval_file_path_2)

    summary_1 = compute_frame_chunks_metrics_summary(
        entities_1, track_sample_man
    )
    summary_2 = compute_frame_chunks_metrics_summary(
        entities_2, track_sample_man
    )
    summary_1.rename(columns=mm.io.motchallenge_metric_names, inplace=True)
    summary_2.rename(columns=mm.io.motchallenge_metric_names, inplace=True)
    summary_diff = summary_2 - summary_1
    summary_diff.sort_index('index', inplace=True)

    if args.output:
        summary_diff.to_csv(args.output)
    
    tab = tabulate(
        summary_diff, headers='keys', tablefmt='fancy_grid', floatfmt='.4f'
    )
    print(tab)

    return 0


if __name__ == '__main__':
    sys.exit(main())
