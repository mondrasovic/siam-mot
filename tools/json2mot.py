import json
import sys

import click


@click.command()
@click.argument('tracking_json_file_path', type=click.Path(exists=True))
@click.argument('output_mot_file_path', type=click.Path())
def main(tracking_json_file_path, output_mot_file_path):
    with open(tracking_json_file_path) as in_file:
        tracking_data = json.load(in_file)

    result_rows = []
    for entity in tracking_data['entities']:
        frame_idx = entity['blob']['frame_idx'] + 1
        x, y, width, height = entity['bb']
        x += 1
        y += 1
        object_id = entity['id'] + 1

        row_data = (
            f'{frame_idx}, {object_id}, {x}, {y}, {width}, {height},'
            ' -1, -1, -1, -1'
        )

        result_rows.append(row_data)

    with open(output_mot_file_path, 'wt') as out_file:
        out_file.write("\n".join(result_rows) + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
