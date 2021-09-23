import collections
import functools

from math import cos

from typing import Deque, Sequence, Tuple

import torch
import numpy as np
import cv2 as cv

from maskrcnn_benchmark.structures.bounding_box import BoxList
from yacs.config import CfgNode
from PIL import Image

from siammot.modelling.reid.model import build_model as build_reid_model
from siammot.modelling.reid.dataset import get_trm as get_reid_trm
from siammot.modelling.reid.config import cfg as cfg_reid
from siammot.modelling.reid.model.baseline import Baseline as ReidBaseline


_instance = None
_reid_config_file_path = "./configs/reid/market_1501.yml"


def cos_sim_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Computes a 2D cosine similarity matrix between two sets of vectors.

    Args:
        a (torch.Tensor): First list of vectors of shape (N, D).
        b (torch.Tensor): Second list of vectors of shape (M, D).
        eps (float, optional): Small value to avoid division by zero.
                               Defaults to 1e-8.

    Returns:
        torch.Tensor: A 2D cosine similarity matrix of shape (N, M).
        The cell at i-th row and j-th column represents the cosine similarity
        between the a[i] and b[j] D-dimensional vectors.
    """
    a_norm = a.norm(dim=1)[..., None]
    b_norm = b.norm(dim=1)[..., None]

    a_ = a / torch.clamp(a_norm, min=eps)
    b_ = b / torch.clamp(b_norm, min=eps)

    sim_matrix = torch.mm(a_, b_.T)

    return sim_matrix


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


class ReIdManager:
    EMB_LRU_CACHE_SIZE = 1024

    def __init__(
        self,
        reid_baseline: ReidBaseline,
        reid_transform,
        max_dormant_frames: int = 100,
    ) -> None:
        assert max_dormant_frames > 0

        self._reid_baseline: ReidBaseline = reid_baseline
        self._reid_transform = reid_transform

        self._frames: Deque[Image.Image] = collections.deque(
            maxlen=max_dormant_frames
        )
        self._curr_frame_idx: int = 0
   
    def add_next_frame(self, frame: Image.Image) -> None:
        self._frames.append(frame)
    
    def increment_frame_idx(self) -> None:
        self._curr_frame_idx += 1

    def preview_current_frame(self) -> None:
        frame = self._get_frame(self._curr_frame_idx)
        img = self._img_pil_to_cv(frame)

        cv.imshow("Preview", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def calc_cosine_sim_matrix(
        self,
        frame_idxs_1: Sequence[int],
        boxes_1: BoxList,
        frame_idxs_2: Sequence[int],
        boxes_2: BoxList
    ) -> torch.Tensor:
        assert (not frame_idxs_1) or (len(boxes_1) == len(frame_idxs_1))
        assert (not frame_idxs_2) or (len(boxes_2) == len(frame_idxs_2))
        assert boxes_1.mode == 'xyxy' and boxes_2.mode == 'xyxy'

        if (len(boxes_1) == 0) or (len(boxes_2) == 0):
            return torch.empty((0, 0))
        
        emb_1 = self._calc_embeddings(frame_idxs_1, boxes_1)
        emb_2 = self._calc_embeddings(frame_idxs_2, boxes_2)

        sim_matrix = cos_sim_matrix(emb_1, emb_2)

        return sim_matrix
    
    def reset(self) -> None:
        self._frames.clear()
        self._curr_frame_idx = 0
    
    def _calc_embeddings(
        self,
        frame_idxs: Sequence[int],
        boxes: BoxList
    ) -> torch.Tensor:
        boxes_int = boxes.bbox.round().int()

        emb = [
            self._calc_embedding(frame_idx, tuple(box_int.tolist()))
            for frame_idx, box_int in zip(frame_idxs, boxes_int)
        ]
        emb = torch.cat(emb, dim=0)

        return emb

    @functools.lru_cache(maxsize=1024)
    def _calc_embedding(
        self,
        frame_idx: int,
        box_int: Tuple[int]
    ) -> torch.Tensor:
        frame = self._get_frame(frame_idx)
        roi = frame.crop(box_int)
        roi = torch.unsqueeze(self._reid_transform(roi), dim=0).cuda()
        emb = self._reid_baseline(roi).cpu().detach()

        return emb

    def _get_frame(self, frame_idx: int) -> Image.Image:
        idx_diff = self._curr_frame_idx - frame_idx
        rel_queue_pos = len(self._frames) - idx_diff - 1
        frame = self._frames[rel_queue_pos]

        return frame
    
    def _img_pil_to_cv(self, img: Image.Image) -> np.ndarray:
        img = np.asarray(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img

    def _img_tensor_to_cv(self, img: torch.Tensor) -> np.ndarray:
        img = img.cpu().detach().squeeze(0).numpy()  # [3, H, W]
        img = img.transpose(1, 2, 0)  # [H, W, 3]
        img = (((img * self._pixel_std) + self._pixel_mean) * 255).round()
        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        return img


def build_or_get_existing_reid_manager(cfg: CfgNode) -> ReIdManager:
    global _instance

    if not _instance:
        global _reid_config_file_path

        cfg_reid.merge_from_file(_reid_config_file_path)
        cfg_reid.freeze()

        baseline = build_reid_model(cfg_reid)
        param_dict = torch.load(cfg_reid.TEST.WEIGHT)
        baseline.load_state_dict(param_dict)
        baseline.cuda()
        baseline.eval()

        transform = get_reid_trm(cfg_reid, is_train=False)

        _instance = ReIdManager(
            baseline,
            transform,
            cfg.MODEL.TRACK_HEAD.MAX_DORMANT_FRAMES
        )
    
    return _instance
