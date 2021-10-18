import sys
import tqdm
import json
import click
import shutil
import pathlib
import functools
import queue
import threading

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
        stages_order = ('input', 'after NMS', 'after ReID', 'output')
        stages = frame_data['stages']

        for stage_name in stages_order:
            stage_data = stages.get(stage_name)
            if stage_data is not None:
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
    start_pt, end_pt = tuple(box[:2]), tuple(box[2:])
    label = f"{entity['id']}"
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


class ClosableQueue(queue.Queue):
    _SENTINEL = object()

    def close(self):
        self.put(self._SENTINEL)

    def __iter__(self):
        while True:
            item = self.get()
            try:
                if item is self._SENTINEL:
                    return
                yield item
            finally:
                self.task_done()


class StageProcessorWorker(threading.Thread):
    def __init__(
        self,
        output_dir,
        w_scale,
        h_scale,
        ignored_stages,
        img_stages_iters_queue,
        imgs_processed_queue
    ):
        super().__init__()

        self.output_dir = output_dir
        self.w_scale = w_scale
        self.h_scale = h_scale
        self.ignored_stages = set(ignored_stages)
        self.img_stages_iters_queue = img_stages_iters_queue
        self.imgs_processed_queue = imgs_processed_queue
    
    def run(self):
        should_resize = not np.allclose(
            np.asarray((self.w_scale, self.h_scale)), 1
        )

        for frame_idx, (img, stages_iter) in self.img_stages_iters_queue:
            for j, (stage_name, entities_iter) in enumerate(
                stages_iter, start=1
            ):
                if stage_name in self.ignored_stages:
                    continue

                curr_img = img.copy()
                for entity in entities_iter:
                    render_entity(curr_img, stage_name, entity)
                
                if should_resize:
                    curr_img = cv.resize(
                        curr_img, None, fx=self.w_scale, fy=self.h_scale
                    )

                img_file_name = f'frame_{frame_idx:04d}_{j:02d}_{stage_name}.jpg'
                img_file_path = str(self.output_dir / img_file_name)
                self.imgs_processed_queue.put((img_file_path, curr_img))
        self.imgs_processed_queue.close()


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
@click.option(
    '-i', '--ignore-stage', multiple=True, help="List of stages to ignore."
)
def main(
    imgs_dir_path,
    output_dir_path,
    debug_dump_file_path,
    w_scale,
    h_scale,
    ignore_stage
):
    output_dir = pathlib.Path(output_dir_path)
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_stages_iters_queue = ClosableQueue()
    imgs_processed_queue = ClosableQueue()
    
    stages_processor_thread = StageProcessorWorker(
        output_dir, w_scale, h_scale, ignore_stage, img_stages_iters_queue,
        imgs_processed_queue
    )
    stages_processor_thread.start()

    data_iter = iter_imgs_and_stages(imgs_dir_path, debug_dump_file_path)
    print("Reading frames...")
    for item in tqdm.tqdm(enumerate(data_iter, start=1)):
        img_stages_iters_queue.put(item)
    img_stages_iters_queue.close()

    print("Saving processed frames...")
    for img_file_path, img in tqdm.tqdm(imgs_processed_queue):
        cv.imwrite(img_file_path, img)
    
    img_stages_iters_queue.join()
    imgs_processed_queue.join()
    stages_processor_thread.join()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
