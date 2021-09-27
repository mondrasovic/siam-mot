import json

from typing import Dict, Sequence, List, Dict, Optional

from maskrcnn_benchmark.structures.bounding_box import BoxList


_instance = None


class TrackSolverDebugger:
    def __init__(self) -> None:
        self._frames: List[Dict] = []
        self._curr_frame: Dict = self._init_new_frame()
        
        self.sample_width: Optional[int] = None
        self.sample_height: Optional[int] = None

    def add_detections(
        self,
        stage: str,
        detections: BoxList,
        active_ids: Sequence[int],
        dormant_ids: Sequence[int]
    ) -> None:
        if (self.sample_width is not None) and (self.sample_height is not None):
            detections = detections.resize(
                (self.sample_width, self.sample_height)
            )
        boxes = detections.bbox.round().int()
        ids = detections.get_field('ids')
        scores = detections.get_field('scores')

        entities = []
        for box, id_, score in zip(boxes, ids, scores):
            id_, score = id_.item(), score.item()

            if id_ in active_ids:
                status = 'active'
            elif id_ in dormant_ids:
                status = 'dormant'
            else:
                status = 'inactive'

            entity = {
                'box': box.tolist(),
                'confidence': score,
                'id': id_,
                'status': status,
            }
            entities.append(entity)
        
        stages = self._curr_frame['stages']
        stages[stage] = {'entities': entities}

    def reset(self) -> None:
        self._frames = []
        self._curr_frame = self._init_new_frame()
    
    def save_frame(self) -> None:
        self._frames.append(self._curr_frame)
        self._curr_frame = self._init_new_frame()
    
    def save_to_file(
        self,
        json_file_path: str
    ) -> None:
        with open(json_file_path, 'wt') as output_file:
            data = {
                'frames': self._frames
            }
            json.dump(data, output_file, indent=2)
    
    @staticmethod
    def _init_new_frame() -> Dict:
        return {'stages': {}}


def build_or_get_existing_track_solver_debugger():
    global _instance

    if _instance is None:
        _instance = TrackSolverDebugger()
    
    return _instance
