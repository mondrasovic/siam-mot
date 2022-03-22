from cProfile import label
import math
import pathlib
import sys

import click
import cv2 as cv
import numpy as np


def mean_image_color(image):
    return tuple(int(c) for c in np.mean(image, axis=(0, 1)).round())


def make_image_square(image):
    height, width, _ = image.shape
    max_side = max(width, height)

    width_diff_half = (max_side - width) / 2
    height_diff_half = (max_side - height) / 2

    top = int(math.ceil(height_diff_half))
    bottom = int(math.floor(height_diff_half))
    left = int(math.ceil(width_diff_half))
    right = int(math.floor(width_diff_half))

    # border_color = mean_image_color(image)
    square_image = cv.copyMakeBorder(
        image, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return square_image, (top, bottom, left, right)


def add_padding_to_bbox_labels(label_rows, border_padding, bbox_start_pos=6):
    left_pos, top_pos, right_pos, bottom_pos = tuple(
        range(bbox_start_pos, bbox_start_pos + 4)
    )
    top_padding, _, left_padding, _ = border_padding

    for label_row in label_rows:
        label_row[left_pos] = str(float(label_row[left_pos]) + left_padding)
        label_row[top_pos] = str(float(label_row[top_pos]) + top_padding)
        label_row[right_pos] = str(float(label_row[right_pos]) + left_padding)
        label_row[bottom_pos] = str(float(label_row[bottom_pos]) + top_padding)


def labels_content_to_rows(labels_content):
    return [line.split() for line in labels_content.splitlines()]


def label_rows_to_content(label_rows):
    return '\n'.join(' '.join(row) for row in label_rows)


@click.command()
@click.argument('input_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
def main(input_dir_path, output_dir_path):
    input_dir = pathlib.Path(input_dir_path)
    output_dir = pathlib.Path(output_dir_path)

    input_images_dir = input_dir / 'image_02'
    input_labels_dir = input_dir / 'label_02'

    for seq_images_dir in input_images_dir.iterdir():
        for image_file in seq_images_dir.iterdir():
            image = cv.imread(str(image_file), cv.IMREAD_COLOR)
            square_image, border_padding = make_image_square(image)
            rel_image_path = image_file.relative_to(input_dir)
            output_file = output_dir / rel_image_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(output_file), square_image)

        seq_name = f'{seq_images_dir.stem}.txt'
        input_labels_file = input_labels_dir / seq_name
        labels_content = input_labels_file.read_text()
        label_rows = labels_content_to_rows(labels_content)
        add_padding_to_bbox_labels(label_rows, border_padding)
        labels_content_new = label_rows_to_content(label_rows)
        rel_label_path = input_labels_file.relative_to(input_dir)
        output_labels_file = output_dir / rel_label_path
        output_labels_file.parent.mkdir(parents=True, exist_ok=True)
        output_labels_file.write_text(labels_content_new)

    return 0


if __name__ == '__main__':
    sys.exit(main())
