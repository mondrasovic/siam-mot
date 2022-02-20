import json
import sys

import click


def mscoco_select_categories_subset(anno_content, valid_category_names):
    valid_category_names = set(valid_category_names)
    orig_categories = anno_content['categories']
    valid_categories = [
        category for category in orig_categories
        if category['name'] in valid_category_names
    ]
    valid_category_ids = set(category['id'] for category in valid_categories)

    orig_annotations = anno_content['annotations']
    valid_annotations = [
        anno for anno in orig_annotations if (anno['iscrowd'] == 0) and
        (anno['category_id'] in valid_category_ids)
    ]

    anno_content['categories'] = valid_categories
    anno_content['annotations'] = valid_annotations


@click.command()
@click.argument('src_anno_file_path', type=click.Path(exists=True))
@click.argument('dst_anno_file_path', type=click.Path())
@click.option(
    '-c', '--categories', multiple=True, help="Selected category names."
)
def main(src_anno_file_path, dst_anno_file_path, categories):
    with open(src_anno_file_path) as in_file:
        anno_content = json.load(in_file)

    mscoco_select_categories_subset(anno_content, categories)

    with open(dst_anno_file_path, 'wt') as out_file:
        json.dump(anno_content, out_file)

    return 0


if __name__ == '__main__':
    sys.exit(main())
