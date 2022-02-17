import argparse
import itertools
import json
import os
import shutil
import sys

import tqdm


def export_coco_dataset(
    dataset_input_dir, dataset_output_dir, dataset_name, subset
):
    dataset_output_dir = os.path.join(dataset_output_dir, dataset_name)
    images_dir_path = os.path.join(dataset_output_dir, 'data')

    if os.path.exists(dataset_output_dir):
        shutil.rmtree(dataset_output_dir)
    os.makedirs(images_dir_path)

    data = {}
    data['info'] = {
        'year': '',
        'version': '',
        'description': 'CrowdHuman dataset',
        'contributor': '',
        'url': '',
        'date_created': ''
    }
    data['licenses'] = []
    person_cat_id = 0
    data['categories'] = [
        {
            'id': person_cat_id,
            'name': 'person',
            'supercategory': 'person'
        }
    ]
    images, annotations = [], []
    data['images'] = images
    data['annotations'] = annotations

    image_id_gen = itertools.count()
    annotation_id_gen = itertools.count()

    image_height, image_width = 540, 960

    tqdm_pbar = tqdm.tqdm(file=sys.stdout)
    with tqdm_pbar as pbar:
        data_iter = iter_image_boxes_pairs(dataset_input_dir, subset)
        for seq_num, image_num, image_file_path, boxes in data_iter:
            pbar.set_description(
                f"processing seq. {seq_num}, sample {image_file_path}"
            )

            dst_file_name = f'{seq_num:02d}_{image_num:04d}.jpg'
            dst_file_path = os.path.join(images_dir_path, dst_file_name)
            shutil.copyfile(image_file_path, dst_file_path)

            image_id = next(image_id_gen)
            image_data = {
                'id': image_id,
                'file_name': dst_file_name,
                'seq_num': seq_num,
                'image_num': image_num,
                'height': image_height,
                'width': image_width,
            }
            images.append(image_data)

            for box in boxes:
                annotation = {
                    'id': next(annotation_id_gen),
                    'image_id': image_id,
                    'category_id': person_cat_id,
                    'bbox': box,
                    'area': box[2] * box[3],
                    'iscrowd': 0,
                }
                annotations.append(annotation)

            pbar.update()

    json_file_path = os.path.join(dataset_output_dir, 'annotations.json')
    with open(json_file_path, 'wt') as out_file:
        json.dump(data, out_file)


def main():
    return 0


if __name__ == '__main__':
    sys.exit(main())
