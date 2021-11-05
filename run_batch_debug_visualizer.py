import sys
import click
import pathlib
import subprocess

import tqdm


def build_dataset_dir_path(datasets_path, subset):
    return (
        pathlib.Path(datasets_path) / 'UA-DETRAC_GluonCV' / 'raw_data' /
        f'Insight-MVT_Annotation_{subset}'
    )


@click.command()
@click.argument('test_dir_path', type=click.Path(exists=True))
@click.option(
    '-s', '--script-path', type=click.Path(exists=True),
    default='./tools/solver_debug_visualizer.py',
    help="Script to visualizer debug info."
)
@click.option(
    '-d', '--datasets-path', type=click.Path(exists=True),
    default='../../datasets', help="Datasets root directory path."
)
@click.option(
    '-o', '--output-dir-path', type=click.Path(), default='../..',
    help="Output directory path."
)
@click.option(
    '--subset', type=click.Choice(['Train', 'Test']), default='Test', 
    show_default=True, help="Data subset type."
)
@click.option(
    '-m', '--model-dir-name', default='DLA-34-FPN_box_EMM_UA_DETRAC', 
    show_default=True, help="Model directory name."
)
@click.option(
    '-p', '--pattern', default='track_solver_debug_*.json', show_default=True, help="Debug dump file pattern."
)
def main(
    test_dir_path,
    script_path,
    datasets_path,
    output_dir_path,
    subset,
    model_dir_name,
    pattern
):
    dataset_dir = build_dataset_dir_path(datasets_path, subset)
    test_dir = pathlib.Path(test_dir_path)
    debug_dir = test_dir / model_dir_name
    output_dir = pathlib.Path(output_dir_path) / 'solver_debug_visualization'

    tqdm_pbar = tqdm.tqdm(file=sys.stdout)
    with tqdm_pbar as pbar:
        for file in debug_dir.glob(pattern):
            sample_name = file.stem[-9:]
            pbar.set_description(f"processing sample {sample_name}")

            curr_out_dir = output_dir / test_dir.stem / sample_name
            if (
                not (
                    curr_out_dir.exists() and
                    any(curr_out_dir.iterdir())
                )
            ):
                curr_imgs_dir = dataset_dir / sample_name

                args = [
                    'python', script_path, str(curr_imgs_dir),
                    str(curr_out_dir), str(file)
                ]
                print()
                subprocess.run(args, shell=True)
            
            pbar.update()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
