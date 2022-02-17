import datetime
import itertools
import json
import os
import shutil
import sys

import click
import tqdm
from PIL import Image


def ensure_empty_dir_exsits(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def get_current_date():
    today = datetime.date.today()
    date = today.strftime(r'%Y/%m/%d')
    return date


def iter_crowdhuman_anno(anno_file_path, box_type='fbox'):
    with open(anno_file_path) as anno_file:
        for anno_entry in map(json.loads, anno_file.readlines()):
            image_id = anno_entry['ID']
            gt_boxes = anno_entry['gtboxes']

            def _iter_boxes():
                for gt_box in gt_boxes:
                    box = gt_box[box_type]
                    yield box

            yield image_id, _iter_boxes()


def get_image_size(image_file_path):
    image = Image.open(image_file_path)
    return image.size


def convert_anno_crowdhuman_to_coco(
    crowdhuman_anno_file_path,
    crowdhuman_images_dir_path,
    coco_output_dir,
    subset_type,
    box_type='fbox'
):
    coco_images_dir_path = os.path.join(coco_output_dir, 'Images')
    coco_anno_dir_path = os.path.join(coco_output_dir, 'annotations')

    ensure_empty_dir_exsits(coco_images_dir_path)
    ensure_empty_dir_exsits(coco_anno_dir_path)

    data = {}
    data['info'] = {
        'year': '',
        'version': '',
        'description': 'CrowdHuman dataset',
        'contributor': '',
        'url': '',
        'date_created': get_current_date()
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
    anno_id_gen = itertools.count()

    tqdm_pbar = tqdm.tqdm(file=sys.stdout)
    with tqdm_pbar as pbar:
        anno_iter = iter_crowdhuman_anno(crowdhuman_anno_file_path, box_type)
        for image_id, boxes_iter in anno_iter:
            pbar.set_description(f"processing image with ID {image_id}")

            image_file_name = f'{image_id}.jpg'
            crowdhuman_image_file_path = os.path.join(
                crowdhuman_images_dir_path, image_file_name
            )
            image_width, image_height = get_image_size(
                crowdhuman_image_file_path
            )

            new_image_id = next(image_id_gen)

            image_data = {
                'id': new_image_id,
                'file_name': image_file_name,
                'height': image_height,
                'width': image_width,
            }
            images.append(image_data)

            for box in boxes_iter:
                annotation = {
                    'id': next(anno_id_gen),
                    'image_id': new_image_id,
                    'category_id': person_cat_id,
                    'bbox': box,
                    'area': box[2] * box[3],
                    'iscrowd': 0,
                }
                annotations.append(annotation)

            pbar.update()

    anno_file_name = f'annotation_{subset_type}_{box_type}.json'
    json_file_path = os.path.join(coco_anno_dir_path, anno_file_name)
    with open(json_file_path, 'wt') as out_file:
        json.dump(data, out_file)


@click.command()
@click.argument('crowdhuman_anno_file_path', type=click.Path(exists=True))
@click.argument('crowdhuman_images_dir_path', type=click.Path(exists=True))
@click.argument('coco_dataset_dir_path', type=click.Path())
@click.option(
    '-b',
    '--box-type',
    type=click.Choice(['fbox', 'vbox']),
    default='fbox',
    show_default=True,
    help="Type of bounding box to load, i.e., (f)ull vs. (v)isible."
)
@click.option(
    '-s',
    '--subset-type',
    type=click.Choice(['train', 'val']),
    default='train',
    show_default=True,
    help="Type of the data subset."
)
def main(
    crowdhuman_anno_file_path, crowdhuman_images_dir_path,
    coco_dataset_dir_path, box_type, subset_type
):
    convert_anno_crowdhuman_to_coco(
        crowdhuman_anno_file_path, crowdhuman_images_dir_path,
        coco_dataset_dir_path, subset_type, box_type
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
