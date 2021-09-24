import sys
import json
import click
import itertools
import pathlib

import cv2 as cv
import numpy as np


def iter_imgs_and_entities(imgs_dir_path, entities):
    def _key_func(entity):
        return entity['blob']['frame_idx']
    
    def _build_converted_entities_iter(entities_orig_iter):
        for entity_orig in entities_orig_iter:
            entity_new = {
                'box': tuple(map(lambda c: int(round(c)), entity_orig['bb'])),
                'id': entity_orig['id'],
                'confidence': entity_orig['confidence']
            }
            yield entity_new
    
    imgs_dir = pathlib.Path(imgs_dir_path)
    files_iter = imgs_dir.iterdir()
    groups_iter = itertools.groupby(entities, key=_key_func)

    for file, (_, entities_orig_iter) in zip(files_iter, groups_iter):
        img = cv.imread(str(file), cv.IMREAD_COLOR)
        entities_iter = _build_converted_entities_iter(entities_orig_iter)
        yield img, entities_iter


def labeled_rectangle(
    img,
    start_pt,
    end_pt,
    label,
    rect_color,
    label_color,
    alpha= 0.85
):
    (x1, y1), (x2, y2) = start_pt, end_pt

    roi = img[y1:y2, x1:x2]
    rect = np.ones_like(roi) * 255
    img[y1:y2, x1:x2] = cv.addWeighted(roi, alpha, rect, 1 - alpha, 0)

    font_face = cv.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    font_thickness = 3

    (text_width, text_height), baseline = cv.getTextSize(
        label, font_face, font_scale, font_thickness)
    text_rect_end = (
        start_pt[0] + text_width, start_pt[1] + text_height + baseline
    )
    cv.rectangle(img, start_pt, text_rect_end, rect_color, -1)
    
    # TODO Somehow calculate the shift.
    text_start_pt = (start_pt[0] + 1, start_pt[1] + text_height + 3)
    cv.putText(
        img, label, text_start_pt, font_face, font_scale, label_color,
        font_thickness, cv.LINE_AA
    )
    cv.putText(
        img, label, text_start_pt, font_face, font_scale, (255, 255, 255),
        max(1, font_thickness - 2), cv.LINE_AA
    )
    cv.rectangle(img, start_pt, end_pt, rect_color, 2, cv.LINE_AA)


def render_entity(img, entity):
    box = entity['box']
    start_pt = box[:2]
    end_pt = (start_pt[0] + box[2], start_pt[1] + box[3])
    label = str(entity['id'])

    labeled_rectangle(img, start_pt, end_pt, label, (0, 255, 0), (0, 0, 255))


@click.command()
@click.argument('imgs_dir_path', type=click.Path())
@click.argument('inference_dump_file_path', type=click.Path())
@click.option(
    '-n', '--start-frame', type=int, default=1, show_default=True,
    help="Frame no. to start from."
)
@click.option(
    '-w', '--win-name', type=str, default="Tracking preview", show_default=True,
    help="Window name."
)
@click.option(
    '--w-scale', type=float, default=1.0, show_default=True,
    help="Width scale factor."
)
@click.option(
    '--h-scale', type=float, default=1.0, show_default=True,
    help="Height scale factor."  
)
def main(
    imgs_dir_path,
    inference_dump_file_path,
    start_frame,
    win_name,
    w_scale,
    h_scale
):
    with open(inference_dump_file_path) as inference_file:
        content = json.load(inference_file)
        entities = content['entities']
    
    imgs_dir = pathlib.Path(imgs_dir_path)

    for i, (img, entities_iter) in enumerate(
        iter_imgs_and_entities(imgs_dir, entities), start=1
    ):
        if i < start_frame:
            continue

        for entity in entities_iter:
            render_entity(img, entity)
        
        img = cv.resize(img, None, fx=w_scale, fy=h_scale)
        cv.imshow(win_name, img)

        key = cv.waitKey(0) & 0xff
        if key == ord('q'):
            break

    cv.destroyWindow(win_name)

    return 0


if __name__ == '__main__':
    sys.exit(main())
