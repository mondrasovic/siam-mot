import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .singleton import Singleton


class ReIdManager(metaclass=Singleton):
    def __init__(self, max_dormant_frames: int) -> None:
        assert max_dormant_frames > 0
    
    def register_curr_frame(self, frame) -> None:
        pass

    def register_curr_detections(self, boxes: BoxList) -> None:
        pass

    def calc_cos_sim_matrix(
        self,
        boxlist_1: BoxList,
        frame_idx_1: int,
        boxlist_2: BoxList,
        frame_idx_2: int
    ) -> torch.tensor:
        raise NotImplementedError  # TODO Implement computing cos. sim.
