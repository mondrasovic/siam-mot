import sys
import pathlib
import os
import shutil

import click
import pandas as pd


def build_latex_table_doc(results_file):
    df = pd.read_csv(results_file, index_col=0)
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

    tokens = results_file.parent.parent.stem.split('_')
    tokens = (t.replace('-', ' - ') for t in tokens)
    caption = f'Siam-MOT Evaluation Report --- {", ".join(tokens)}'
    table = df.to_latex(bold_rows=True, caption=caption)
    content = r'''\documentclass{article}

\usepackage[a4paper,landscape,width=260mm,top=15mm,bottom=15mm]{geometry}

\usepackage[utf8]{inputenc}
\usepackage{booktabs}

\pagenumbering{gobble}

\begin{document}
''' + table + r'''
\end{document}
'''
    return content


@click.command()
@click.option(
    '-p', '--results-patt', default='eval_results.csv', show_default=True,
    help="Results file name pattern."
)
@click.option(
    '--report-file', default='results_table_report.pdf', show_default=True,
    help="PDF report file name.")
def main(results_patt, report_file):
    search_patt = f'./{results_patt}'
    for results_file in pathlib.Path('.').rglob(search_patt):
        curr_dir = results_file.parent
        tmp_dir = curr_dir / '_tmp_latex_dir'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        latex_file_path = tmp_dir / '_tmp_tex_src.tex'
        latex_content = build_latex_table_doc(results_file)
        latex_file_path.write_text(latex_content)

        prev_cwd = os.getcwd()
        os.chdir(str(tmp_dir))
        os.system(f'pdflatex {latex_file_path.stem}')
        os.chdir(prev_cwd)

        shutil.move(
            str(tmp_dir / f'{latex_file_path.stem}.pdf'),
            str(curr_dir / report_file)
        )
        shutil.rmtree(str(tmp_dir))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
