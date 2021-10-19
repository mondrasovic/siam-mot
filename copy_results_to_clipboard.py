import sys
import click

import pandas as pd


@click.command()
@click.argument('results_csv_file_path', type=click.Path(exists=True))
def main(results_csv_file_path):
    df = pd.read_csv(results_csv_file_path, index_col=0)
    df.rename(
        columns={
            'num_frames': 'frames',
            'num_matches': 'matches',
            'num_switches': 'switches',
            'num_false_positives': 'FP',
            'num_misses': 'FN',
            'num_objects': 'objs',
            'num_predictions': 'preds',
            'num_fragmentations': 'fragms',
            'mostly_tracked': 'MT',
            'partially_tracked': 'PT',
            'precision': 'prec',
            'recall': 'rec',
            'idf1': 'IDF1',
            'mota': 'MOTA',
            'motp': 'MOTP'
        }, inplace=True
    )
    pd.io.clipboards.to_clipboard(df, excel=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
