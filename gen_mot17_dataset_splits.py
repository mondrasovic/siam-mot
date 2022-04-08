import json
import pathlib
import sys

import click


def extract_seq_num(seq_name):
    return int(seq_name.split('-')[1])


@click.command()
@click.argument('splits_json_file_path', type=click.Path(exists=True))
def main(splits_json_file_path):
    with open(splits_json_file_path) as in_file:
        splits = json.load(in_file)

    train_seqs = sorted(splits['train'])
    all_seq_nums = set(map(extract_seq_num, train_seqs))

    output_dir = pathlib.Path(splits_json_file_path).parent

    def _save_splits_file(seq_names):
        seq_nums = set(map(extract_seq_num, seq_names))
        seq_nums_str = '_'.join(
            f'{seq_num:02d}' for seq_num in sorted(seq_nums)
        )
        splits_file_path = str(output_dir / f'splits_{seq_nums_str}.json')
        with open(splits_file_path, 'wt') as out_file:
            curr_splits = {'train': seq_names, 'test': []}
            json.dump(curr_splits, out_file, indent=2)

    for held_out_seq_num in all_seq_nums:
        curr_train_splits, curr_test_splits = [], []

        for seq_name in train_seqs:
            seq_num = extract_seq_num(seq_name)
            if seq_num == held_out_seq_num:
                curr_test_splits.append(seq_name)
            else:
                curr_train_splits.append(seq_name)

        _save_splits_file(curr_train_splits)
        _save_splits_file(curr_test_splits)

    return 0


if __name__ == '__main__':
    sys.exit(main())
