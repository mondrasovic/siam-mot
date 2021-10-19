# Ingestion script for KITTI-MOT
# (http://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset.

# Explanation of attributes:
# https://github.com/pratikac/kitti/blob/master/readme.tracking.txt

import os
import sys
import tqdm
import argparse
import pathlib

from datetime import datetime

from gluoncv.torch.data.gluoncv_motion_dataset.dataset import (
    AnnoEntity, DataSample, FieldNames, GluonCVMotionDataset, SplitNames
)
from gluoncv.torch.data.gluoncv_motion_dataset.utils.ingestion_utils import \
    process_dataset_splits


_RELEVANT_CLASSES = {'Car', 'Van', 'Truck', 'Tram'}
_CLASS_LABELS = dict(
    (vt, i) for i, vt in enumerate(_RELEVANT_CLASSES, start=1)
)


def sample_from_txt(txt_file_path, split_dir_name, args):
    txt_file = pathlib.Path(txt_file_path)
    sample_name = txt_file.stem
    sample = DataSample(id=sample_name)

    n_frames = 0
    for tokens in map(str.split, txt_file.read_text().splitlines()):
        frame_idx = int(tokens[0])
        n_frames = max(n_frames, frame_idx + 1)
        
        obj_class = tokens[2]
        if obj_class not in _RELEVANT_CLASSES:
            continue

        obj_id = int(tokens[1])
        time_ms = int(round((frame_idx / args.fps) * 1000))

        entity = AnnoEntity(time=time_ms, id=obj_id)
        entity.confidence = float(tokens[-1])

        x1, y1, x2, y2 = tuple(map(float, tokens[6:10]))
        entity.bbox = [round(x1), round(y1), round(x2 - x1), round(y2 - y1)] 
                    
        entity.blob = {
            'frame_idx':   frame_idx,
            'obj_class':   obj_class,
            'sample_name': sample_name,
        }
        entity.labels = {obj_class: _CLASS_LABELS[obj_class]}

        sample.add_entity(entity)
    
    # Need to replace the Windows path separator by UNIX-like to make the path
    # working across different platforms. Linux struggles with mixing path
    # separators whereas Windows does not.
    rel_data_path = os.path.join(
        split_dir_name, 'image_02', sample_name
    ).replace('\\', '/')
    sample.metadata = {
        FieldNames.DATA_PATH:  rel_data_path,
        FieldNames.FPS:        args.fps,
        FieldNames.NUM_FRAMES: n_frames,
        FieldNames.RESOLUTION: {
            'width': args.img_width, 'height': args.img_height,
        },
    }

    return sample


def ingest_kitti_mot(args):
    dataset = GluonCVMotionDataset(
        annotation_file='anno.json', root_path=args.dataset_dir_path,
        load_anno=False
    )
    dataset.metadata = {
        FieldNames.DESCRIPTION:   "KITTI-MOT benchmark dataset ingestion",
        FieldNames.DATE_MODIFIED: str(datetime.now()),
    }

    dataset_anno_dir = (
        pathlib.Path(args.dataset_dir_path) / GluonCVMotionDataset.DATA_DIR /
        'training' / 'label_02'
    )

    tqdm_pbar = tqdm.tqdm(file=sys.stdout)
    with tqdm_pbar as pbar:
        for sample_txt_file_path in map(str, dataset_anno_dir.iterdir()):
            pbar.set_description(f"reading sample {sample_txt_file_path}")
            sample = sample_from_txt(
                sample_txt_file_path, 'training', args
            )
            dataset.add_sample(sample)
            pbar.update()

    dataset.dump()

    return dataset


def write_data_split(dataset):
    def split_func(sample):
        data_path = sample.data_relative_path

        if 'training' in data_path:
            return SplitNames.TRAIN
        elif 'testing' in data_path:
            return SplitNames.TEST
        
        raise RuntimeError("unrecognized data split")
    
    process_dataset_splits(dataset, split_func, save=True)


def main():
    parser = argparse.ArgumentParser(
        description="KITTI-MOT dataset text annotations ingestion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dataset_dir_path', type=str,
        help="Root directory path to the dataset."
    )
    parser.add_argument(
        '--fps', type=int, default=10, help="FPS for all data samples."
    )
    parser.add_argument(
        '--img-width', type=int, default=1382, help="Image width."
    )
    parser.add_argument(
        '--img-height', type=int, default=512, help="Image height"
    )
    args = parser.parse_args()

    dataset = ingest_kitti_mot(args)
    write_data_split(dataset)

    return 0


if __name__ == '__main__':
    sys.exit(main())
