import os
import sys
import json
import click
import shutil
import dataclasses
import subprocess

from typing import Iterable, List


@dataclasses.dataclass(frozen=True)
class CfgOptSpec:
    name: str
    alias: str
    val: str


def normalize_path(path: str) -> str:
    return path.replace('\\', '/')


def build_model_path(train_dir_path: str, model_suffix: str) -> str:
    file_name = f'model_{model_suffix}.pth'
    model_path = os.path.join(train_dir_path, file_name)
    model_path = normalize_path(model_path)

    return model_path


def build_output_dir_path(
    output_root_path: str, dataset_name: str, model_suffix: str,
    cfg_opts: Iterable[str]
) -> str:
    if dataset_name == 'UA_DETRAC':
        dataset_shortcut = 'uadt'
    else:
        raise ValueError('unrecognized dataset name')
    
    tokens = [dataset_shortcut]
    tokens.extend(f'{c.alias}-{c.val}' for c in cfg_opts)
    tokens.append(model_suffix)

    output_dir_name = '_'.join(tokens)
    output_dir_path = os.path.join(output_root_path, output_dir_name)
    output_dir_path = normalize_path(output_dir_path)

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
        'python3', '-m', 'tools.test_net',
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
    output_root_path: str,
    csv_file_name: str,
    model_suffixes: Iterable[str],
    cfg_opts: Iterable[CfgOptSpec],
    rm_existing_output: bool = True
) -> List[str]:
    for model_suffix in model_suffixes:
        for cfg_opt in cfg_opts:
            model_file_path = build_model_path(train_dir_path, model_suffix)
            output_dir_path = build_output_dir_path(
                output_root_path, dataset_name, model_suffix, cfg_opt
            )
            if rm_existing_output and os.path.exists(output_dir_path):
                shutil.rmtree(output_dir_path)
            
            csv_file_path = os.path.join(output_dir_path, csv_file_name)
            csv_file_path = normalize_path(csv_file_path)

            cmd = build_run_test_cmd(
                config_file_path, model_file_path, dataset_name, data_subset,
                csv_file_path, output_dir_path, cfg_opt
            )
            yield cmd


@click.command()
@click.argument('param_json_file_path', type=click.Path(exists=True))
def main(param_json_file_path: click.Path) -> int:
    with open(param_json_file_path, 'rt') as file_handle:
        params = json.load(file_handle)
    
    train_dir_path = params['train_dir_path']
    config_file_path = params['config_file_path']
    dataset_name = params['dataset_name']
    data_subset = params['data_subset']
    output_root_path = params['output_root_path']
    csv_file_name = params['csv_file_name']
    model_suffixes = params['model_suffixes']
    cfg_opts = [[CfgOptSpec(*c) for c in g] for g in params['cfg_opts_groups']]

    for cmd in iter_cmd_args(
        train_dir_path, config_file_path, dataset_name, data_subset,
        output_root_path, csv_file_name, model_suffixes, cfg_opts
    ):
        cmd_str = " ".join(cmd)
        print(cmd_str + "\n")
        
        # print(f"Running command:\n{cmd_str}\n{'-' * 80}\n")
        # subprocess.call(cmd)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
