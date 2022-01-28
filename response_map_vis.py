import math
import pathlib
import sys

import click
import cv2 as cv
import numpy as np


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


def stack_response_heatmaps(
    heatmaps, direction='horizontal', *, out_width=None, out_height=None
):
    if direction == 'horizontal':
        stack_func = np.hstack
    elif direction == 'vertical':
        stack_func = np.vstack
    else:
        raise ValueError(f"unsupported stacking direction {direction}")

    if (out_width is None) and (out_height is None):
        heatmaps_resized = heatmaps
    else:
        heatmaps_resized = []

        for heatmap in heatmaps:
            height, width, _ = heatmap.shape

            new_width = width if out_width is None else out_width
            new_height = height if out_height is None else out_height

            heatmap_resized = cv.resize(
                heatmap, (new_width, new_height),
                interpolation=cv.INTER_LANCZOS4
            )
            heatmaps_resized.append(heatmap_resized)

    heatmaps_stacked = stack_func(heatmaps_resized)

    return heatmaps_stacked


def draw_heatmaps_comparison(
    image,
    boxes_1,
    boxes_2,
    heatmaps_1,
    heatmaps_2,
    *,
    color_1=(89, 141, 252),
    color_2=(96, 207, 145),
    background=(0, 0, 0)
):
    max_width = max(image.shape[1], heatmaps_1.shape[1], heatmaps_2.shape[1])
    frame_parts = []

    for frame_part in (heatmaps_1, heatmaps_2, image):
        _, width, _ = frame_part.shape

        width_diff_half = (max_width - width) / 2
        left_px = int(math.ceil(width_diff_half))
        right_px = int(math.floor(width_diff_half))

        frame_part = cv.copyMakeBorder(
            frame_part,
            top=0,
            bottom=0,
            left=left_px,
            right=right_px,
            borderType=cv.BORDER_CONSTANT,
            value=background
        )
        frame_parts.append(frame_part)

    for boxes, color in zip((boxes_1, boxes_2), (color_1, color_2)):
        for box in boxes:
            draw_labeled_rectangle(
                image,
                tuple(box[:2]),
                tuple(box[2:]),
                '',
                rect_color=color,
                label_color=(255, 255, 255)
            )

    frame = np.vstack(frame_parts)
    return frame


def iter_response_maps_and_boxes(vis_dir_path, seq_name):
    data_dir = pathlib.Path(vis_dir_path) / seq_name
    response_maps_boxes_iter = iter(
        sorted(list(data_dir.iterdir()), key=lambda f: f.stem)
    )

    while True:
        try:
            response_map_file = next(response_maps_boxes_iter)
            boxes_file = next(response_maps_boxes_iter)
        except StopIteration:
            break
        else:
            response_map = np.load(response_map_file)
            boxes = np.load(boxes_file)
            yield response_map, boxes


def iter_images(dataset_dir_path, seq_name):
    images_dir = (
        pathlib.Path(dataset_dir_path) / 'raw_data' /
        'Insight-MVT_Annotation_Test' / seq_name
    )
    images_list = sorted(list(images_dir.iterdir()), key=lambda f: f.stem)
    for image_file in images_list:
        image = cv.imread(str(image_file), cv.IMREAD_COLOR)
        yield image


def resize_boxes(boxes_xyxy, old_size, new_size):
    scale_xy = np.asarray(new_size) / np.asarray(old_size)
    scale = np.concatenate((scale_xy, scale_xy))
    boxes_resized = boxes_xyxy * scale

    return boxes_resized


@click.command()
@click.argument('dataset_dir_path', type=click.Path(exists=True))
@click.argument('vis_dir_path', type=click.Path(exists=True))
@click.argument('seq_id')
def main(dataset_dir_path, vis_dir_path, seq_id):
    seq_name = 'MVI_' + seq_id

    for image, (response_map, boxes) in zip(
        iter_images(dataset_dir_path, seq_name),
        iter_response_maps_and_boxes(vis_dir_path, seq_name)
    ):
        print(image.shape, response_map.shape, boxes.shape)
        cv.imshow("Preview", image)

    return 0


if __name__ == '__main__':
    sys.exit(main())
