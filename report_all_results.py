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
        },
        inplace=True
    )

    model_dir = results_file.parent.parent
    attention_dir = model_dir.parent
    dataset_dir = attention_dir.parent

    model = model_dir.stem
    attention = {
        'attention_inc': 'FC Attention',
        'no_attention': 'None',
        'with_dsa': 'Deformable Siamese Attention',
        'without_dsa': 'None'
    }[attention_dir.stem]
    # model_dir = results_file.parent.parent
    # solver_dir = model_dir.parent
    # loss_dir = solver_dir.parent
    # dataset_dir = loss_dir.parent

    # model = model_dir.stem
    # solver = {'orig': 'Original', 'fNMS': 'Feature-NMS'}[solver_dir.stem]
    # loss = {'none': 'None', 'contr': 'Contrastive', 'tripl': 'Triplet'}[
    #     loss_dir.stem
    # ]
    dataset = {'uadt': 'UA-DETRAC'}[dataset_dir.stem]

    caption = (
        'Siam-MOT Evaluation Report --- ' +
        f'Dataset: {dataset}, Attention: {attention}, Model: {model}'
    )
    # caption = (
    #     'Siam-MOT Evaluation Report --- ' +
    #     f'Dataset: {dataset}, Embedding Loss: {loss}, ' +
    #     f'Solver: {solver}, Model: {model}'
    # )
    table = df.to_latex(bold_rows=True, caption=caption)
    content = r'''\documentclass{article}

\usepackage[a4paper,landscape,width=260mm,top=15mm,bottom=15mm]{geometry}

\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage[labelfont=bf]{caption}

\pagenumbering{gobble}

\begin{document}
''' + table + r'''
\end{document}
'''
    return content


@click.command()
@click.option(
    '-p',
    '--results-patt',
    default='eval_results.csv',
    show_default=True,
    help="Results file name pattern."
)
@click.option(
    '--report-file',
    default='results_table_report.pdf',
    show_default=True,
    help="PDF report file name."
)
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

        src_pdf_file = tmp_dir / f'{latex_file_path.stem}.pdf'
        dst_pdf_file = curr_dir / report_file

        if dst_pdf_file.exists():
            dst_pdf_file.unlink()
        shutil.move(str(src_pdf_file), str(dst_pdf_file))
        shutil.rmtree(str(tmp_dir))

    return 0


if __name__ == '__main__':
    sys.exit(main())
