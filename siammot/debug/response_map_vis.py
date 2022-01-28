import itertools
import os
import pathlib
from typing import List

import numpy as np
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList

_instance = None


class ResponseMapVisualizer:
    def __init__(self, output_dir_path: str = './response_map_vis') -> None:
        self.output_dir_path: str = output_dir_path
        self.curr_seq_name: str = None
        self.frame_num_iter = None

    def add_response_map(
        self, response_map: torch.Tensor, track_result: List[BoxList]
    ) -> None:
        response_map = response_map.detach().cpu().numpy()
        curr_frame_num = next(self.frame_num_iter)
        frame_text = f"frame_{curr_frame_num:04d}"
        self._save_numpy_file(frame_text + "_response_map.npy", response_map)

        for box_list in track_result:
            boxes = box_list.bbox.detach().cpu().numpy()
            self._save_numpy_file(frame_text + "_boxes.npy", boxes)

    def init_new_sequence(self, seq_name: str) -> None:
        self.curr_seq_name = seq_name
        self.frame_num_iter = itertools.count(1)

    def _save_numpy_file(self, file_name: str, content: np.ndarray) -> None:
        dir_path = os.path.join(self.output_dir_path, self.curr_seq_name)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, file_name)

        np.save(file_path, content)


def build_or_get_existing_response_map_visualizer():
    global _instance

    if _instance is None:
        _instance = ResponseMapVisualizer()

    return _instance
