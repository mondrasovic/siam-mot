import collections

from typing import Deque, Sequence

import torch
import numpy as np
import cv2 as cv

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import ImageList
from yacs.config import CfgNode

from siammot.modelling.reid.singleton import Singleton


def draw_box(
    img: np.ndarray,
    box: np.ndarray,
    color=(0, 255, 0),
    thickness=3
) -> None:
    pt1, pt2 = tuple(box[:2]), tuple(box[2:])
    cv.rectangle(
        img,
        pt1,
        pt2,
        color=color,
        thickness=thickness,
        lineType=cv.LINE_AA
    )


class ReIdManager(metaclass=Singleton):
    def __init__(
        self,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
        max_dormant_frames: int
    ) -> None:
        assert max_dormant_frames > 0

        self._pixel_mean = pixel_mean
        self._pixel_std = pixel_std

        self._frames: Deque[torch.Tensor] = collections.deque(
            maxlen=max_dormant_frames
        )
        self._boxes: Deque[BoxList] = collections.deque(
            maxlen=max_dormant_frames
        )
    
    def add_next_frame(self, frame: ImageList) -> None:
        assert frame.tensors.ndim == 4
        assert len(frame.tensors) == 1
        assert len(frame.image_sizes) == 1

        self._frames.append(frame.tensors)

    def add_next_boxes(self, boxes: BoxList) -> None:
        self._boxes.append(boxes)
    
    def preview_current_frame(self) -> None:
        frame = self._frames[-1]
        frame = self._img_tensor_to_cv(frame)

        boxes = self._boxes[-1].bbox.cpu().detach().numpy()
        boxes = boxes.round().astype(np.int)

        for box in boxes:
            draw_box(frame, box)
        
        cv.imshow("Preview", frame)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def calc_cos_sim_matrix(
        self
    ) -> torch.Tensor:
        raise NotImplementedError  # TODO Implement computing cos. sim.
    
    def _img_tensor_to_cv(self, img: torch.Tensor) -> np.ndarray:
        assert img.ndim == 4
        assert len(img) == 1

        img = img.cpu().detach().squeeze(0).numpy()  # [3, H, W]
        img = img.transpose(1, 2, 0)  # [H, W, 3]
        img = (((img * self._pixel_std) + self._pixel_mean) * 255).round()
        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        return img


def build_reid_manager(cfg: CfgNode) -> ReIdManager:
    reid_man = ReIdManager(
        cfg.INPUT.PIXEL_MEAN,
        cfg.INPUT.PIXEL_STD,
        cfg.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES
    )
    return reid_man
