import os
import sys
import click
import shutil
import dataclasses
import subprocess
import itertools

from typing import Iterable, Sequence, List, Tuple


@dataclasses.dataclass(frozen=True)
class CfgOptSpec:
    name: str
    alias: str
    val: str


def build_model_path(train_dir_path: str, model_suffix: str) -> str:
    file_name = f'model_{model_suffix}.pth'
    return os.path.join(train_dir_path, file_name)


def build_output_dir_path(
    dataset_name: str, model_suffix: str, cfg_opts: Iterable[str]
) -> str:
    if dataset_name == 'UA_DETRAC':
        dataset_shortcut = 'uadt'
    else:
        raise ValueError('unrecognized dataset name')
    
    tokens = [dataset_shortcut]
    tokens.extend(f'{c.alias}-{c.val}' for c in cfg_opts)
    tokens.append(model_suffix)

    output_dir_name = '_'.join(tokens)
    output_dir_path = os.path.join('.', output_dir_name)

    return output_dir_path


def build_run_test_cmd(
    config_file_path: str,
    model_file_path: str,
    dataset_name: str,
    data_subset: str,
    csv_file_path: str,
    output_dir_path: str,
    cfg_opts: Iterable[CfgOptSpec]
):
    run_test_args = [
        'python', '-m', 'tools.test_net',
        '--config-file', config_file_path,
        '--model-file', model_file_path,
        '--test-dataset', dataset_name,
        '--set', data_subset,
        '--eval-csv-file', csv_file_path,
        '--output-dir', output_dir_path,
    ]
    for cfg_opt in cfg_opts:
        run_test_args.append(cfg_opt.name)
        run_test_args.append(cfg_opt.val)

    return run_test_args


def iter_cmd_args(
    train_dir_path: str,
    config_file_path: str,
    dataset_name: str,
    data_subset: str,
    csv_file_name: str,
    model_suffixes: Iterable[str],
    cfg_opts: Iterable[CfgOptSpec]
) -> List[str]:
    for model_suffix in model_suffixes:
        for cfg_opt in cfg_opts:
            model_file_path = build_model_path(train_dir_path, model_suffix)
            output_dir_path = build_output_dir_path(
                dataset_name, model_suffix, cfg_opt
            )
            csv_file_path = os.path.join(output_dir_path, csv_file_name)

            cmd = build_run_test_cmd(
                config_file_path, model_file_path, dataset_name, data_subset,
                csv_file_path, output_dir_path, cfg_opt
            )
            yield cmd


@click.command()
@click.argument('train_dir_path', type=click.Path(exists=True))
@click.argument('config_file_path', type=click.Path(exists=True))
@click.option(
    '-d', '--dataset', default='UA_DETRAC', show_default=True,
    help="Dataset name."
)
@click.option(
    '-s', '--subset', default='test', show_default=True, help="Data subset."
)
@click.option(
    '--csv-file-name', default='eval_results.csv', show_default=True,
    help="Evaluation results CSV file name."
)
def main(
    train_dir_path: click.Path,
    config_file_path: click.Path,
    dataset: str,
    subset: str,
    csv_file_name: str,
) -> int:
    model_suffixes = (
        'final', '0080000', '0070000', '0060000', '0050000', '0040000',
        '0030000',
    )
    cmd_arg_specs = (
        (
            CfgOptSpec(
                'MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS', 'loss', 'contrastive'
            ),
            CfgOptSpec('MODEL.TRACK_HEAD.SOLVER_TYPE', 'slr', 'feature_emb')
        ),
        (
            CfgOptSpec(
                'MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS', 'loss', 'contrastive'
            ),
            CfgOptSpec('MODEL.TRACK_HEAD.SOLVER_TYPE', 'slr', 'original')
        ),
        (
            CfgOptSpec(
                'MODEL.TRACK_HEAD.EMM.FEATURE_EMB_LOSS', 'loss', 'original'
            ),
            CfgOptSpec('MODEL.TRACK_HEAD.SOLVER_TYPE', 'slr', 'original')
        ),
    )

    for cmd in iter_cmd_args(
        train_dir_path, config_file_path, dataset, subset, csv_file_name,
        model_suffixes, cmd_arg_specs
    ):
        cmd_str = " ".join(cmd)
        print(f"Running command:\n{cmd_str}\n{'-' * 80}\n")

        # subprocess.call(cmd, text=True, shell=True)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
