import argparse
import os
import sys

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from siammot.configs.defaults import cfg
from siammot.modelling.rcnn import build_siammot
from siammot.data.adapters.augmentation.build_augmentation import build_siam_augmentation
from siammot.modelling.track_head.EMM.attention import DeformableSiameseAttention
from siammot.modelling.track_head.EMM.xcorr import xcorr_depthwise


def draw_heatmaps_on_image(
    image: np.ndarray,
    heatmaps: np.ndarray,
    boxlist: BoxList,
    alpha: float = 0.4
):
    assert 0 < alpha < 1, "alpha must be in (0, 1) interval"

    # if len(heatmaps) != len(boxlist):
    #     raise ValueError("the number of heatmaps and boxes must be equal")

    boxlist.clip_to_image()
    boxlist = boxlist.convert('xyxy')
    boxes = boxlist.bbox.numpy().round().astype(np.int)
    heatmap_image = np.zeros_like(image)

    for heatmap, (x1, y1, x2, y2) in zip(heatmaps, boxes):
        heatmap_width = x2 - x1
        heatmap_height = y2 - y1

        heatmap_resized = cv.resize(
            heatmap, (heatmap_width, heatmap_height),
            interpolation=cv.INTER_LANCZOS4
        )
        heatmap_image[y1:y2, x1:x2] = heatmap_resized

    image_blend = cv.addWeighted(image, alpha, heatmap_image, 1 - alpha, 0)

    return image_blend


def draw_labeled_rectangle(
    image, start_point, end_point, label, rect_color, label_color, alpha=0.85
):
    (x1, y1), (x2, y2) = start_point, end_point

    roi = image[y1:y2, x1:x2]
    rect = np.ones_like(roi) * 255
    image[y1:y2, x1:x2] = cv.addWeighted(roi, alpha, rect, 1 - alpha, 0)

    font_face = cv.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 0.8
    font_thickness = 1

    (text_width, text_height
    ), baseline = cv.getTextSize(label, font_face, font_scale, font_thickness)
    text_rect_end = (
        start_point[0] + text_width, start_point[1] + text_height + baseline
    )
    cv.rectangle(image, start_point, text_rect_end, rect_color, thickness=-1)

    text_start_pt = (start_point[0] + 1, start_point[1] + text_height + 3)
    cv.putText(
        image,
        label,
        text_start_pt,
        font_face,
        font_scale,
        label_color,
        thickness=font_thickness,
        lineType=cv.LINE_AA
    )
    cv.putText(
        image,
        label,
        text_start_pt,
        font_face,
        font_scale, (255, 255, 255),
        thickness=max(1, font_thickness - 2),
        lineType=cv.LINE_AA
    )
    cv.rectangle(
        image,
        start_point,
        end_point,
        rect_color,
        thickness=2,
        lineType=cv.LINE_AA
    )


def draw_boxlist(image, boxlist, color):
    boxes = boxlist.convert('xyxy').bbox.cpu().numpy().round().astype(np.int)
    for x1, y1, x2, y2 in boxes:
        draw_labeled_rectangle(
            image, (x1, y1), (x2, y2), "", color, (255, 255, 255)
        )


def create_response_heatmaps(response_maps, reduction='mean'):
    if reduction == 'mean':
        reduce_func = np.mean
    elif reduction == 'max':
        reduce_func = np.max
    else:
        raise ValueError(f"unsupported reduction {reduction}")

    if response_maps.ndim == 3:
        response_maps = np.expand_dims(response_maps, axis=0)

    response_maps_reduced = reduce_func(response_maps, axis=1, keepdims=True)
    response_maps_reduced = np.transpose(response_maps_reduced, (0, 2, 3, 1))

    min_val = np.min(response_maps_reduced)
    max_val = np.max(response_maps_reduced)

    response_maps_norm = (
        ((response_maps_reduced - min_val) / (max_val - min_val)) * 255
    ).round().astype(np.uint8)
    response_heatmaps = [
        cv.applyColorMap(r, cv.COLORMAP_JET) for r in response_maps_norm
    ]

    return response_heatmaps


def parse_args():
    parser = argparse.ArgumentParser(
        description="SiamMOT response map visualization."
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar='FILE',
        help="path to config file",
        type=str
    )
    parser.add_argument(
        "--model-file",
        default=None,
        metavar='MODEL',
        help="path to model file",
        type=str
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        metavar='N',
        help="local process rank"
    )
    parser.add_argument(
        'exemplar_image_file',
        metavar='EXEMPLAR',
        help="exemplar image file path"
    )
    parser.add_argument(
        'search_image_file', metavar='SEARCH', help="search image file path"
    )
    parser.add_argument(
        'opts',
        help="overwriting the training config from commandline",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    return args


class SiamMOTImageInferencer:
    def __init__(self, cfg, device, model_file=None) -> None:
        self.device = device
        self.model = build_siammot(cfg).to(self.device)

        checkpointer = DetectronCheckpointer(
            cfg, self.model, save_dir=model_file
        )
        if os.path.isfile(model_file):
            checkpointer.load(model_file)
        elif os.path.isdir(model_file):
            checkpointer.load(use_latest=True)
        else:
            raise KeyError("No checkpoint is found")

        self.transform = build_siam_augmentation(cfg, is_train=False)
        self.dummy_bbox = torch.tensor([[0, 0, 1, 1]])

        self.model.eval()

    def predict_on_file(self, image_file_path, to_cpu=True) -> BoxList:
        image = Image.open(image_file_path)
        return self.predict(image, to_cpu)

    def predict(self, image, to_cpu=True) -> BoxList:
        image_size = image.size
        image_width, image_height = image_size

        dummy_target = BoxList(self.dummy_bbox, image_size, mode='xywh')
        image_tensor, _ = self.transform(image, dummy_target)
        image_tensor = image_tensor.to(self.device)
        boxlist = self.model(image_tensor)[0]

        boxlist.resize([image_width, image_height]).convert('xywh')

        if to_cpu:
            boxlist = boxlist.to(torch.device('cpu'))

        return boxlist


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.MODEL.DEVICE)
    torch.cuda.set_device(args.local_rank)

    inferencer = SiamMOTImageInferencer(cfg, device, args.model_file)
    response_map = None

    def hook_fn(module, input, output):
        if isinstance(module, DeformableSiameseAttention):
            nonlocal response_map
            attentional_template_features, attentional_sr_features = output
            response_map = xcorr_depthwise(
                attentional_sr_features, attentional_template_features
            )

        return None

    torch.nn.modules.module.register_module_forward_hook(hook_fn)

    with torch.no_grad():
        exemplar_boxlist = inferencer.predict_on_file(args.exemplar_image_file)
        search_boxlist = inferencer.predict_on_file(args.search_image_file)

    iou_thresh = 0.9
    box_iou = boxlist_iou(exemplar_boxlist, search_boxlist)
    above_thresh_mask = box_iou > iou_thresh
    idxs = above_thresh_mask.nonzero()[:, 0].unique()

    print(idxs)
    image = cv.imread(args.search_image_file)
    # heatmaps = create_response_heatmaps(response_map.cpu().numpy())
    # image_heatmap_blend = draw_heatmaps_on_image(
    #     image, heatmaps, exemplar_boxlist
    # )
    draw_boxlist(image, exemplar_boxlist, (0, 0, 255))
    draw_boxlist(image, search_boxlist, (0, 255, 0))
    cv.imwrite('response_maps_vis.png', image)

    return 0


if __name__ == '__main__':
    sys.exit(main())
