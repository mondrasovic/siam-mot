import click
import json
import os
import pathlib
import sys

from typing import Sequence, Tuple, List, Dict, Iterator, Optional, Iterable

import cv2 as cv
import numpy as np
import tqdm

from shapely import geometry


PointT = Tuple[int, int]
PolygonT = Sequence[PointT]


def convex_hull(points) -> np.ndarray:
    return np.squeeze(cv.convexHull(np.asarray(points)))


def draw_filled_polygon(img, points, color) -> None:
    cv.drawContours(
        img, [points], 0, color=color, thickness=-1, lineType=cv.LINE_AA
    )


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


class InvalidPolygon(Exception):
    pass


class AnnotationsManager:
    def __init__(self, json_file_path: str) -> None:
        self._json_file_path: str = json_file_path
        self._anno_map: Dict[str, PolygonT] = None

    def __getitem__(self, sample_name: str) -> PolygonT:
        return self._anno_map[sample_name]
    
    def __setitem__(self, sample_name, polygon) -> None:
        if polygon is not None:
            self._anno_map[sample_name] = polygon

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
    
    def iter_samples(self) -> Iterator[Tuple[str, PolygonT]]:
        return iter(self._anno_map.items())
    
    def iter_existing_sample_names(
        self,
        sample_names: Sequence[str]
    ) -> Iterator[Tuple[str, PolygonT]]:
        sample_names = set(sample_names)
        return iter(
            (name, polygon)
            for name, polygon in self._anno_map.items()
            if name in sample_names
        )


class PolygonAnnotator:
    _SLEEP_MS: int = 10
    _POINT_SIZE_COEF: float = 0.01

    def __init__(
        self,
        win_name: str = "Polygon Annotation Preview",
        *,
        save_key: str = 's',
        clear_key: str = 'c',
        quit_key: str = 'q'
    ) -> None:
        self._assure_key_is_valid(save_key, "save")
        self._assure_key_is_valid(clear_key, "clear")
        self._assure_key_is_valid(quit_key, "quit")

        self.save_key: str = save_key
        self.clear_key: str = clear_key
        self.quit_key: str = quit_key

        self.win_name =(
            f"{win_name} | {self.save_key} - save, "
            f"{self.clear_key} - clear, {self.quit_key} - quit"
        )

    def run_interactive(self, img_file_path: str) -> List[PointT]:
        img_orig = cv.imread(img_file_path)
        point_size = int(round(min(img_orig.shape[:2]) * self._POINT_SIZE_COEF))
        roi_color = img_mean_color(img_orig)
        selected_points = []

        def _mouse_click_event(event, x, y, flags, params):
            if event != cv.EVENT_LBUTTONDOWN:
                return
            
            selected_points = params
            selected_points.append((x, y))

        cv.namedWindow(self.win_name)
        cv.setMouseCallback(self.win_name, _mouse_click_event, selected_points)

        exit_loop = False

        while True:
            curr_img = img_orig.copy()

            def _draw_point(point, size, color):
                cv.circle(curr_img, point, size, color, -1, cv.LINE_AA)

            if len(selected_points) > 2:
                points_hull = convex_hull(selected_points)
                draw_filled_polygon(curr_img, points_hull, roi_color)

            for point in selected_points:
                _draw_point(point, point_size * 2, (0, 0, 0))
                _draw_point(point, point_size, (255, 255, 255))
            
            cv.imshow(self.win_name, curr_img)
            
            key = cv.waitKey(self._SLEEP_MS) & 0xff
            if key == ord(self.quit_key):
                exit_loop = True
                break
            elif key == ord(self.save_key):
                break
            elif key == ord(self.clear_key):
                selected_points.clear()
        
        if cv.getWindowProperty(self.win_name, cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow(self.win_name)
        
        if len(selected_points) < 3:
            if not exit_loop:
                raise InvalidPolygon()
            ret_points = None
        else:
            ret_points = convex_hull(selected_points)
            ret_points = [list(map(int, point)) for point in ret_points]

        return exit_loop, ret_points
    
    @staticmethod
    def _assure_key_is_valid(key: str, name: str) -> None:
        assert len(key) == 1, f"{name} key must be of length 1"
        assert ord('a') <= ord(key) <= ord('z'), f"{name} key must be from a-z."


def create_annotations(
    dataset_root_path: str,
    sample_names: Iterable[str],
    anno_man: AnnotationsManager,
    update: bool = False,
    verbose: bool = False
) -> None:
    annotator = PolygonAnnotator()

    exit_loop = False

    for sample_name, img_file_path in iter_data_samples(
        dataset_root_path, sample_names
    ):
        if exit_loop:
            if verbose:
                print("exiting interactive mode on user request")
            break

        if (not update) and (sample_name in anno_man):
            continue

        if verbose:
            print(f"creating annotations for sample {sample_name}")
        
        while True:
            try:
                exit_loop, polygon = annotator.run_interactive(img_file_path)
            except InvalidPolygon:
                if verbose:
                    print(f"invalid polygon provided, try again")
            else:
                anno_man[sample_name] = polygon
                break


def read_json(file_path):
    with open(file_path, 'rt') as file_handle:
        content = json.load(file_handle)
        return content


def write_json(data, file_path, *, indent=None):
    with open(file_path, 'wt') as file_handle:
        json.dump(data, file_handle, indent=indent)


def draw_polygon_anno(img, polygon):
    color = img_mean_color(img)
    draw_filled_polygon(img, np.asarray(polygon), color)


def build_entities_occl_overlap_filter(polygon, overlap_thresh=0.5):
    polygon_shape = geometry.Polygon(polygon)

    def _filter_entity(entity):
        x, y, w, h = entity['bb']
        box = geometry.box(x, y, x + w, y + h)
        box_area = w * h
        isect_area = polygon_shape.intersection(box).area
        overlap_ratio = isect_area / box_area

        return overlap_ratio < overlap_thresh
    
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
   
    for sample_name, polygon in samples_iter:
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
            draw_polygon_anno(img, polygon)

            img_rel_path = src_img_file.relative_to(src_root_dir)
            dst_img_file = dst_root_dir / img_rel_path
            dst_img_file.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(dst_img_file), img)
        
        entities_filter = build_entities_occl_overlap_filter(
            polygon, overlap_thresh
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
