import sys
import tqdm
import json
import click
import shutil
import pathlib
import functools

import cv2 as cv
import numpy as np


def iter_imgs_and_stages(imgs_dir_path, debug_dump_file_path):
    imgs_dir = pathlib.Path(imgs_dir_path)
    
    with open(debug_dump_file_path, 'rt') as debug_file:
        content = json.load(debug_file)
        frames_data = content['frames']
    
    def _build_entities_iter(stage_data):
        status_vals = ('inactive', 'dormant', 'active')
        
        def _sort_by_status_key(entity):
            return status_vals.index(entity['status'])
        
        yield from iter(sorted(stage_data['entities'], key=_sort_by_status_key))
    
    def _build_stages_iter(frame_data):
        stages_order = ('input', 'after NMS', 'output')
        stages = frame_data['stages']

        for stage_name in stages_order:
            stage_data = stages[stage_name]
            entities_iter = _build_entities_iter(stage_data)
            yield stage_name, entities_iter
    
    for file, frame_data in zip(imgs_dir.iterdir(), frames_data):
        img = cv.imread(str(file), cv.IMREAD_COLOR)
        stages_iter = _build_stages_iter(frame_data)
        yield img, stages_iter


def labeled_rectangle(
    img,
    start_pt,
    end_pt,
    label,
    rect_color,
    label_color,
    alpha=0.85
):
    (x1, y1), (x2, y2) = start_pt, end_pt

    roi = img[y1:y2, x1:x2]
    rect = np.ones_like(roi) * 255
    img[y1:y2, x1:x2] = cv.addWeighted(roi, alpha, rect, 1 - alpha, 0)

    font_face = cv.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1.5
    font_thickness = 2

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


def render_entity(img, stage_name, entity):
    box = entity['box']
    start_pt, end_pt = box[:2], box[2:]
    label = f"{entity['id']}"
    status = entity['status']

    label_color = (200, 200, 200)

    if status == 'active':
        rect_color = (0, 255, 0)
    elif status == 'dormant':
        rect_color = (255, 0, 0)
    else:
        rect_color = (0, 0, 255)
    
    render_stage_text = functools.partial(
        cv.putText, img=img, text=f"Stage: {stage_name}", org=(10, 50),
        fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL, fontScale=3, lineType=cv.LINE_AA
    )
    render_stage_text(color=(0, 0, 0), thickness=5)
    render_stage_text(color=(255, 255, 255), thickness=3)

    labeled_rectangle(img, start_pt, end_pt, label, rect_color, label_color)


@click.command()
@click.argument('imgs_dir_path', type=click.Path())
@click.argument('output_dir_path', type=click.Path())
@click.argument('debug_dump_file_path', type=click.Path())
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
    output_dir_path,
    debug_dump_file_path,
    w_scale,
    h_scale
):
    output_dir = pathlib.Path(output_dir_path)
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_iter = iter_imgs_and_stages(imgs_dir_path, debug_dump_file_path)
    for i, (img, stages_iter) in tqdm.tqdm(enumerate(data_iter, start=1)):
        if i > 20:
            break
        for j, (stage_name, entities_iter) in enumerate(stages_iter, start=1):
            curr_img = img.copy()
            for entity in entities_iter:
                render_entity(curr_img, stage_name, entity)
            
            curr_img = cv.resize(curr_img, None, fx=w_scale, fy=h_scale)

            img_file_name = f"frame_{i:04d}_{j:02d}_{stage_name}.jpg"
            img_file_path = str(output_dir / img_file_name)
            cv.imwrite(img_file_path, curr_img)

    return 0


if __name__ == '__main__':
    sys.exit(main())
