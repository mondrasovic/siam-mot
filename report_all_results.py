import sys
import pathlib

import click
import pandas as pd


@click.command()
@click.option(
    '-p', '--results-patt', default='eval_results.csv', show_default=True,
    help="Results file name pattern."
)
def main(results_patt):
    search_patt = f'./{results_patt}'
    for results_file in pathlib.Path('.').rglob(search_patt):
        
    return 0


if __name__ == '__main__':
    sys.exit(main())
