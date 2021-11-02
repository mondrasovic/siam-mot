import click
import json
import os
import pathlib
import sys

from typing import (
    Callable, Sequence, Tuple, List, Dict, Iterator, Optional, Iterable
)

import cv2 as cv
import numpy as np
import tqdm


PointT = Tuple[int, int]
BoxT = Tuple[PointT, PointT, PointT, PointT]
BoxesT = Sequence[BoxT]



def img_mean_color(img: np.ndarray) -> Tuple[int, int, int]:
    return tuple(int(round(c)) for c in np.mean(img, axis=(0, 1)))


def iter_data_samples(
    dataset_root_path: str, sample_names: Optional[Sequence[str]] = None
) -> Iterator[Tuple[str, str]]:
    if sample_names:
        sample_names = set(sample_names)
    
    dataset_dir = pathlib.Path(dataset_root_path)
    pattern = './raw_data/Insight-MVT_Annotation_T*/MVI_*'

    for sample_dir in dataset_dir.rglob(pattern):
        sample_name = sample_dir.stem
        if sample_names and (sample_name not in sample_names):
            continue

        first_file = next(iter(sample_dir.iterdir()))
        img_file_path = str(first_file)

        yield sample_name, img_file_path


class AnnotationsManager:
    def __init__(self, json_file_path: str) -> None:
        self._json_file_path: str = json_file_path
        self._anno_map: Dict[str, BoxesT] = None

    def __getitem__(self, sample_name: str) -> BoxesT:
        return self._anno_map[sample_name]
    
    def __setitem__(self, sample_name, boxes) -> None:
        if boxes is not None:
            self._anno_map[sample_name] = boxes

    def __contains__(self, sample_name) -> bool:
        return sample_name in self._anno_map
    
    def __enter__(self):
        if os.path.exists(self._json_file_path):
            with open(self._json_file_path, 'rt') as file_handle:
                self._anno_map = json.load(file_handle)
        else:
            self._anno_map = {}
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            with open(self._json_file_path, 'wt') as file_handle:
                json.dump(self._anno_map, file_handle, indent=2)
    
    def iter_samples(self) -> Iterator[Tuple[str, BoxesT]]:
        return iter(self._anno_map.items())
    
    def iter_existing_sample_names(
        self,
        sample_names: Sequence[str]
    ) -> Iterator[Tuple[str, BoxesT]]:
        sample_names = set(sample_names)
        return iter(
            (name, polygon)
            for name, polygon in self._anno_map.items()
            if name in sample_names
        )


class PolygonAnnotator:
    def __init__(
        self,
        win_name: str = "Polygon Annotation Preview",
        *,
        show_crosshair: bool = False,
        from_center: bool = False,
    ) -> None:
        self.win_name =(
            f"{win_name} | 'enter' - next selection, 'esc' - quit"
        )

        self.show_crosshair: bool = show_crosshair
        self.from_center: bool = from_center

    def run_interactive(self, img_file_path: str) -> BoxesT:
        img = cv.imread(img_file_path)
        boxes = cv.selectROIs(
            self.win_name, img, showCrosshair=self.show_crosshair,
            fromCenter=self.from_center
        )
        if len(boxes) > 0:
            boxes = boxes.tolist()

        if cv.getWindowProperty(self.win_name, cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow(self.win_name)
        
        return boxes
    

def create_annotations(
    dataset_root_path: str,
    sample_names: Iterable[str],
    anno_man: AnnotationsManager,
    *,
    update: bool = False,
    verbose: bool = False
) -> None:
    annotator = PolygonAnnotator()

    for sample_name, img_file_path in iter_data_samples(
        dataset_root_path, sample_names
    ):
        if (not update) and (sample_name in anno_man):
            continue

        if verbose:
            print(f"creating annotations for sample {sample_name}")
        
        boxes = annotator.run_interactive(img_file_path)
        exit_loop = len(boxes) == 0
        if exit_loop:
            if verbose:
                print("exiting interactive mode on user request")
            break
        anno_man[sample_name] = boxes


def read_json(file_path):
    with open(file_path, 'rt') as file_handle:
        content = json.load(file_handle)
        return content


def write_json(data, file_path, *, indent=None):
    with open(file_path, 'wt') as file_handle:
        json.dump(data, file_handle, indent=indent)


def draw_boxes_anno(img: np.ndarray, boxes: BoxesT) -> None:
    color = img_mean_color(img)

    for box in boxes:
        pt1 = tuple(box[:2])
        pt2 = (box[0] + box[2], box[1] + box[3])

        cv.rectangle(
            img, pt1, pt2, color=color, thickness=-1, lineType=cv.LINE_AA
        )


def intersection_over_area(
        box: np.ndarray,
        boxes: np.ndarray,
        eps: float = 1e-8
    ) -> np.ndarray:
        assert (box.ndim == 2) and (box.shape[0] == 1)
        assert (boxes.ndim == 2) and (boxes.shape[1] == 4)
        assert (box[:, :2] <= box[:, 2:]).all()
        assert (boxes[:, :2] <= boxes[:, 2:]).all()

        coords_tl = np.maximum(boxes[:, :2], box[:, :2])
        coords_br = np.minimum(boxes[:, 2:], box[:, 2:])
        
        wh = (coords_br - coords_tl).clip(min=0)
        intersect_areas = wh[:, 0] * wh[:, 1]
        box_area = np.prod(box[0, 2:] - box[0, :2])
        intersect_ratios = intersect_areas.astype(np.float) / (box_area + eps)

        return intersect_ratios


def build_entities_occl_overlap_filter(
    boxes: BoxesT,
    overlap_thresh: float = 0.5
) -> Callable[[Dict], bool]:
    boxes = np.asfarray(boxes)
    boxes[:, 2:] += boxes[:, :2]

    def _filter_entity(entity: Dict) -> bool:
        x, y, w, h = entity['bb']
        box = np.asfarray((x, y, x + w, y + h))[None, ...]
        overlap_ratio = intersection_over_area(box, boxes)

        return (overlap_ratio < overlap_thresh).all()
    
    return _filter_entity


def generate_modified_dataset(
    input_root_path: str,
    output_root_path: str,
    sample_names: Iterable[str],
    anno_man: AnnotationsManager,
    *,
    overlap_thresh: float = 0.5,
    verbose: bool = False
) -> None:
    if verbose:
        print("generating augmented dataset with new annotations")
    
    src_root_dir = pathlib.Path(input_root_path)
    dst_root_dir = pathlib.Path(output_root_path)

    src_anno_dir = src_root_dir / 'annotation'
    src_anno_file_path = str(src_anno_dir / 'anno.json')
    src_splits_file_path = str(src_anno_dir / 'splits.json')
    anno_data = read_json(src_anno_file_path)
    src_splits_data = read_json(src_splits_file_path)

    test_sample_names = set(src_splits_data['test'])

    anno_samples_data = anno_data['samples']
    dst_splits_data = {'train': [], 'test': []}

    src_data_dir = src_root_dir / 'raw_data'

    if sample_names:
        samples_iter = anno_man.iter_existing_sample_names(sample_names)
    else:
        samples_iter = anno_man.iter_samples()
   
    for sample_name, boxes in samples_iter:
        if verbose:
            print(f"processing sample: {sample_name}")
        
        subset_name = 'test' if sample_name in test_sample_names else 'train'

        dst_splits_data[subset_name].append(sample_name)

        subset_dir_name = 'Insight-MVT_Annotation_' + subset_name.capitalize()
        src_sample_dir = src_data_dir / subset_dir_name / sample_name

        dir_iter = src_sample_dir.iterdir()
        if verbose:
            dir_iter = tqdm.tqdm(dir_iter)
        
        for src_img_file in dir_iter:
            img = cv.imread(str(src_img_file), cv.IMREAD_COLOR)
            draw_boxes_anno(img, boxes)

            img_rel_path = src_img_file.relative_to(src_root_dir)
            dst_img_file = dst_root_dir / img_rel_path
            dst_img_file.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(dst_img_file), img)
        
        entities_filter = build_entities_occl_overlap_filter(
            boxes, overlap_thresh
        )
        anno_samples_data[sample_name]['entities'] = list(
            filter(entities_filter, anno_samples_data[sample_name]['entities'])
        )

    dst_anno_dir = dst_root_dir / 'annotation'
    dst_anno_dir.mkdir(parents=True, exist_ok=True)
    dst_anno_file_path = str(dst_anno_dir / 'anno.json')
    dst_splits_file_path = str(dst_anno_dir / 'splits.json')
    write_json(anno_data, dst_anno_file_path)
    write_json(dst_splits_data, dst_splits_file_path, indent=2)


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('sample_names', nargs=-1)
@click.option(
    '-o', '--output-dir', type=click.Path(),
    help="Output directory. If set, then the modified dataset is generated."
)
@click.option(
    '-t', '--thresh',
    type=click.FloatRange(0, 1, min_open=True, max_open=True), default=0.5,
    show_default=True,
    help="Overlap threshold for removing occluded annotations."
)
@click.option(
    '--anno-file', default='occlusion_anno.json', show_default=True,
    help="File name containing occlusion annotations."
)
@click.option(
    '-a', '--annotate', is_flag=True,
    help="Runs annotation mode prior to generating the dataset."
)
@click.option(
    '-u', '--update', is_flag=True, help="Update existing annotations."
)
@click.option('-v', '--verbose', is_flag=True, help="Enables verbose mode.")
def main(
    input_dir,
    sample_names,
    output_dir,
    thresh,
    anno_file,
    annotate,
    update,
    verbose
) -> int:
    """Dataset modifier for imputing polygon-defined occlusion into GluonCV-like
    dataset.
    """
    anno_file_path = os.path.join(input_dir, anno_file)

    if annotate:
        with AnnotationsManager(anno_file_path) as anno_man:
            create_annotations(
                input_dir, sample_names, anno_man, update=update,
                verbose=verbose
            )
    
    if output_dir:
        with AnnotationsManager(anno_file_path) as anno_man:
            generate_modified_dataset(
                input_dir, output_dir, sample_names, anno_man,
                overlap_thresh=thresh, verbose=verbose
            )

    return 0


if __name__ == '__main__':
    sys.exit(main())
