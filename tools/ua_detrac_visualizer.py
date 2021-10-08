import os
import sys
import tqdm
import click
import shutil
import random
import pathlib
import functools
import dataclasses

from xml.etree import ElementTree
from typing import Any, Iterator, Sequence, Tuple, cast, Callable

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


BBoxT = Tuple[int, int, int, int]
ColorT = Tuple[int, int, int]

_N_CLASSES = 13
_WIN_NAME = "Frame preview"


@dataclasses.dataclass(frozen=True)
class DataEntity:
    obj_id: int
    box: BBoxT
    vehicle_type: str


EntityRendererT = Callable[[np.ndarray, DataEntity], None]


@dataclasses.dataclass(frozen=True)
class SampleFrame:
    frame_num: int
    time_ms: int
    img_file_path: str
    entities: Sequence[DataEntity]
    ignored_regions: Sequence[BBoxT]


def init_colors(n_colors: int, randomize: bool = False) -> Sequence[ColorT]:
    color_map = plt.cm.get_cmap('Spectral', n_colors)
    colors = [
        tuple(int(round(c * 255)) for c in color_map(i)[:3])
        for i in range(n_colors)
    ]

    if randomize:
        random.shuffle(colors)
    
    return cast(Sequence[ColorT], colors)


# TODO Refactor drawing "nice" text.

def transparent_rectangle(img, start_pt, end_pt, alpha=0.5):
    (x1, y1), (x2, y2) = start_pt, end_pt

    roi = img[y1:y2, x1:x2]
    rect = np.ones_like(roi) * 255
    img[y1:y2, x1:x2] = cv.addWeighted(roi, alpha, rect, 1 - alpha, 0)


def labeled_rectangle(
    img,
    start_pt,
    end_pt,
    label,
    rect_color,
    label_color,
    alpha=0.85
):
    transparent_rectangle(img, start_pt, end_pt, alpha)

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
    render_text = functools.partial(
        cv.putText, img=img, text=label, org=text_start_pt,
        fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, lineType=cv.LINE_AA
    )
    render_text(color=label_color, thickness=font_thickness)
    render_text(color=(255, 255, 255), thickness=max(1, font_thickness - 2))

    cv.rectangle(img, start_pt, end_pt, rect_color, 2, cv.LINE_AA)


def build_entity_renderer(
    n_colors: int,
    label_color: ColorT = (200, 200, 200)
) -> EntityRendererT:
    colors = init_colors(n_colors)
    color_map = {}
    color_idx = 0

    def _render_entity(img: np.ndarray, entity: DataEntity) -> None:
        nonlocal color_idx

        box = entity.box
        start_pt = entity.box[:2]
        end_pt = (start_pt[0] + box[2], start_pt[1] + box[3])
        label = str(entity.obj_id)
        vehicle_type = entity.vehicle_type
        
        rect_color = color_map.get(vehicle_type)
        if rect_color is None:
            rect_color = colors[color_idx % len(colors)]
            color_map[vehicle_type] = rect_color
            color_idx += 1
        
        labeled_rectangle(img, start_pt, end_pt, label, rect_color, label_color)
    
    return _render_entity


def build_time_str(time_ms: int) -> str:
    msec_in_min = 1000 * 60
    n_mins = time_ms // msec_in_min
    time_ms %= msec_in_min
    n_secs = time_ms // 1000
    time_ms %= 1000

    return f'{n_mins:02d}:{n_secs:02d}.{time_ms}'


def read_and_render_frame(
    frame: SampleFrame,
    entity_renderer: EntityRendererT
) -> np.ndarray:
    img = cv.imread(frame.img_file_path, cv.IMREAD_COLOR)

    for ignored_region in frame.ignored_regions:
        start_pt = ignored_region[:2]
        end_pt = (
            start_pt[0] + ignored_region[2], start_pt[1] + ignored_region[1]
        )
        transparent_rectangle(img, start_pt, end_pt, alpha=0.6)
    
    for entity in frame.entities:
        entity_renderer(img, entity)
    
    time_str = build_time_str(frame.time_ms)
    headline = f"Frame #{frame.frame_num:04d} | time {time_str}"
    render_headline_text = functools.partial(
        cv.putText, img=img, text=headline, org=(10, 50),
        fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, lineType=cv.LINE_AA
    )
    render_headline_text(color=(0, 0, 0), thickness=3)
    render_headline_text(color=(255, 255, 255), thickness=2)

    return img


def deduce_xml_and_imgs_paths(
    dataset_dir_path: str,
    sample_name: str
) -> Tuple[str, str]:
    dataset_dir = pathlib.Path(dataset_dir_path)
    test_dir = dataset_dir / 'Insight-MVT_Annotation_Test'
    test_sample_names = (d.stem for d in test_dir.iterdir() if d.is_dir())
    suffix = 'Test' if sample_name in test_sample_names else 'Train'
    xml_file_path = str(
        dataset_dir / 'DETRAC_public' / ('540p-' + suffix) /
        f'{sample_name}_v3.xml'
    )
    imgs_dir_path = str(
        dataset_dir / ('Insight-MVT_Annotation_' + suffix) / sample_name
    )

    return xml_file_path, imgs_dir_path


def iter_sample_frames_from_xml(
    xml_file_path: str,
    imgs_dir_path: str,
    fps: int = 25
) -> Iterator[SampleFrame]:
    def _read_box(node_attr) -> BBoxT:
        def _coord(x: Any) -> int:
            return int(round(float(x)))
        
        x = _coord(node_attr['left'])
        y = _coord(node_attr['top'])
        w = _coord(node_attr['width'])
        h = _coord(node_attr['height'])

        return (x, y, w, h)

    tree = ElementTree.parse(xml_file_path)
    root = tree.getroot()

    ignored_regions = []
    for box in root.findall('./ignored_region/box'):
        box = _read_box(box.attrib)
        ignored_regions.append(box)
    
    for frame in root.findall('./frame'):
        frame_num = int(frame.attrib['num'])
        time_ms = int(round(((frame_num - 1) / fps) * 1000))
        img_file_name = f'img{frame_num:05d}.jpg'
        img_file_path = os.path.join(imgs_dir_path, img_file_name)

        entities = []
        for target in frame.findall('.//target'):
            obj_id = int(target.attrib['id'])

            box_attr = target.find('box').attrib
            box = _read_box(box_attr)

            attrib_attr = target.find('attribute').attrib
            vehicle_type = attrib_attr['vehicle_type']
            
            entity = DataEntity(obj_id, box, vehicle_type)
            entities.append(entity)
        
        sample_frame = SampleFrame(
            frame_num, time_ms, img_file_path, entities, ignored_regions
        )
        yield sample_frame


@click.command()
@click.argument('dataset_dir_path', type=click.Path(exists=True))
@click.argument('sample_name', type=str)
@click.argument('output_dir_path', type=click.Path())
@click.option('-s', '--show', is_flag=True, help="Shows rendered frames.")
def main(
    dataset_dir_path: click.Path,
    sample_name: str,
    output_dir_path: click.Path,
    show: bool
) -> int:
    output_dir = pathlib.Path(output_dir_path)
    if output_dir.exists():
        shutil.rmtree(output_dir_path, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_file_path, imgs_dir_path = deduce_xml_and_imgs_paths(
        dataset_dir_path, sample_name
    )

    entity_renderer = build_entity_renderer(_N_CLASSES)

    for sample_frame in tqdm.tqdm(iter_sample_frames_from_xml(
        xml_file_path, imgs_dir_path
    )):
        img = read_and_render_frame(sample_frame, entity_renderer)
        img_file_name = pathlib.Path(sample_frame.img_file_path).name
        img_file_path = str(output_dir / img_file_name)

        cv.imwrite(img_file_path, img)
        if show:
            cv.imshow(_WIN_NAME, img)
            if (cv.waitKey(0) & 0xff) == ord('q'):
                break
    cv.destroyWindow(_WIN_NAME)

    return 0


if __name__ == '__main__':
    sys.exit(main())
