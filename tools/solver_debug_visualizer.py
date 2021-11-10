import sys
import json
import shutil
import pathlib
import functools
import multiprocessing

import click
import cv2 as cv
import numpy as np


def iter_img_files_and_stages(imgs_dir_path, debug_dump_file_path):
    imgs_dir = pathlib.Path(imgs_dir_path)
    
    with open(debug_dump_file_path, 'rt') as debug_file:
        content = json.load(debug_file)
        frames_data = content['frames']
    
    def _read_entities(stage_data):
        status_vals = ('inactive', 'dormant', 'active')
    
        return sorted(
            stage_data['entities'],
            key=lambda e: status_vals.index(e['status'])
        )
    
    def _read_stages(frame_data):
        stages_order = ('input', 'after NMS', 'after ReID', 'output')
        stages = frame_data['stages']

        for stage_name in stages_order:
            stage_data = stages.get(stage_name)
            if stage_data is not None:
                entities = _read_entities(stage_data)
                yield stage_name, entities
    
    for file, frame_data in zip(imgs_dir.iterdir(), frames_data):
        img_file_path = str(file)
        stages_data = tuple(_read_stages(frame_data))
        yield img_file_path, stages_data


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
    font_scale = 0.8
    font_thickness = 1

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
    start_pt, end_pt = tuple(box[:2]), tuple(box[2:])
    label = f"{entity['id']}:{entity['confidence']:.1%}"
    status = entity['status']

    if status == 'active':
        rect_color = (80, 204, 0)
    elif status == 'dormant':
        rect_color = (255, 51, 51)
    else:
        rect_color = (0, 0, 153)
    
    render_stage_text = functools.partial(
        cv.putText, img=img, text=f"Stage: {stage_name}", org=(10, 50),
        fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL, fontScale=3, lineType=cv.LINE_AA
    )
    render_stage_text(color=(0, 0, 0), thickness=5)
    render_stage_text(color=(255, 255, 255), thickness=3)
    labeled_rectangle(
        img, start_pt, end_pt, label, rect_color, label_color=(200, 200, 200)
    )


def process_frame(
    frame_idx,
    img_file_path,
    stages_data,
    output_dir,
    stage_names
):
    img_orig = cv.imread(img_file_path, cv.IMREAD_COLOR)

    for j, (stage_name, entities_data) in enumerate(stages_data, start=1):
        if stage_name in stage_names:
            curr_img = img_orig.copy()

            for entity in entities_data:
                render_entity(curr_img, stage_name, entity)
            
            img_file_name = f'frame_{frame_idx:04d}_{j:02d}_{stage_name}.jpg'
            img_file_path = str(output_dir / img_file_name)
            cv.imwrite(img_file_path, curr_img)


@click.command()
@click.argument('imgs_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
@click.argument('debug_dump_file_path', type=click.Path(exists=True))
@click.argument('stages', nargs=-1)
def main(imgs_dir_path, output_dir_path, debug_dump_file_path, stages):
    output_dir = pathlib.Path(output_dir_path)
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    n_workers = min(multiprocessing.cpu_count(), 4)
    with multiprocessing.Pool(n_workers) as pool:
        data_iter = iter_img_files_and_stages(
            imgs_dir_path, debug_dump_file_path
        )
        stage_names = set(stages)
        args_iter = (
            (i, img_file_path, stages_data, output_dir, stage_names)
            for i, (img_file_path, stages_data) in enumerate(data_iter, start=1)
        )
        pool.starmap(process_frame, args_iter, chunksize=2)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
