import os
import sys
import json
import click
import pathlib
import dataclasses
from itertools import cycle, groupby
from typing import Iterable, Dict, Optional
from nbformat import write

import numpy as np


@dataclasses.dataclass(frozen=True)
class CfgOptSpec:
    name: str
    val: str


def normalize_path(path: str) -> str:
    return path.replace('\\', '/')


def build_model_path(train_dir_path: str, model_suffix: str) -> str:
    file_name = f'model_{model_suffix}.pth'
    model_path = os.path.join(train_dir_path, file_name)
    model_path = normalize_path(model_path)

    return model_path


def build_output_dir_path(
    output_root_path: str,
    dataset_name: str,
    model_suffix: str,
    cfg_opts: Iterable[CfgOptSpec],
    cfg_val_map: Optional[Dict[str, str]] = None
) -> str:
    if dataset_name == 'UA_DETRAC':
        dataset_shortcut = 'uadt'
    elif 'MOT' in dataset_name:
        dataset_shortcut = 'mot'
    else:
        raise ValueError('unrecognized dataset name')

    output_dir = pathlib.Path(output_root_path)
    output_dir /= dataset_shortcut
    for cfg_opt in cfg_opts:
        if cfg_val_map:
            dir_name = cfg_val_map.get(cfg_opt.val, cfg_opt.val)
        else:
            dir_name = cfg_opt.val
        output_dir /= dir_name
    output_dir /= model_suffix

    output_dir_path = normalize_path(str(output_dir))

    return output_dir_path


def build_run_test_cmd(
    local_rank: int, config_file_path: str, model_file_path: str,
    dataset_name: str, data_subset: str, csv_file_name: str,
    output_dir_path: str, cfg_opts: Iterable[CfgOptSpec]
):
    run_test_args = [
        'python3', '-m', 'tools.test_net', '--local_rank',
        str(local_rank), '--config-file', config_file_path, '--model-file',
        model_file_path, '--test-dataset', dataset_name, '--set', data_subset,
        '--eval-csv-file', csv_file_name, '--output-dir', output_dir_path,
        "MODEL.DEVICE", f"\"cuda:{local_rank}\""
    ]
    for cfg_opt in cfg_opts:
        run_test_args.append(cfg_opt.name)
        run_test_args.append(cfg_opt.val)

    return run_test_args


def iter_cmd_args(
    train_dir_path: str,
    local_ranks: Iterable[int],
    config_file_path: str,
    dataset_name: str,
    data_subset: str,
    output_root_path: str,
    csv_file_name: str,
    model_suffixes: Iterable[str],
    cfg_opts: Iterable[CfgOptSpec],
    cfg_val_map: Optional[Dict[str, str]] = None
) -> str:
    local_ranks_iter = cycle(local_ranks)
    for cfg_opt in cfg_opts:
        for model_suffix in model_suffixes:
            model_file_path = build_model_path(train_dir_path, model_suffix)
            output_dir_path = build_output_dir_path(
                output_root_path, dataset_name, model_suffix, cfg_opt,
                cfg_val_map
            )
            local_rank = next(local_ranks_iter)

            cmd = build_run_test_cmd(
                local_rank, config_file_path, model_file_path, dataset_name,
                data_subset, csv_file_name, output_dir_path, cfg_opt
            )
            yield ' '.join(cmd)


def get_cmd_cuda_device(cmd):
    cuda_str = "cuda:"
    pos = cmd.find(cuda_str)

    if pos < 0:
        device_id = 0
    else:
        id_pos = pos + len(cuda_str)
        device_id = int(cmd[id_pos:id_pos + 1])

    return device_id


def write_cmds_device_group_to_files(
    cmds: Iterable[str], n_out_files: int, file_name_format: str,
    start_file_id: int
):
    n_cmds = len(cmds)
    n_out_files = min(n_out_files, n_cmds)
    idxs = np.linspace(0, n_cmds + 1, n_out_files + 1).astype(np.int)

    idxs_iter = enumerate(zip(idxs[:-1], idxs[1:]), start=start_file_id)
    for i, (start, end) in idxs_iter:
        with open(file_name_format.format(i), 'wt') as out_file:
            out_file.write("\n\n".join(cmds[start:end]) + "\n")


@click.command()
@click.argument('param_json_file_path', type=click.Path(exists=True))
@click.option(
    '-n',
    '--n-out-files',
    type=int,
    default=6,
    show_default=True,
    help="Number of output files."
)
@click.option(
    '-f',
    '--file-name-format',
    default='run_eval_{}.sh',
    show_default=True,
    help="Script file name format."
)
def main(
    param_json_file_path: click.Path, n_out_files: int, file_name_format: str
) -> int:
    with open(param_json_file_path, 'rt') as file_handle:
        params = json.load(file_handle)

    train_dir_path = params['train_dir_path']
    local_ranks = params.get('local_ranks', [0])
    config_file_path = params['config_file_path']
    dataset_name = params['dataset_name']
    data_subset = params['data_subset']
    output_root_path = params['output_root_path']
    csv_file_name = params['csv_file_name']
    model_suffixes = params['model_suffixes']
    cfg_opts = [[CfgOptSpec(*c) for c in g] for g in params['cfg_opts_groups']]
    cfg_val_map = params.get('cfg_val_map')

    cmds = list(
        iter_cmd_args(
            train_dir_path, local_ranks, config_file_path, dataset_name,
            data_subset, output_root_path, csv_file_name, model_suffixes,
            cfg_opts, cfg_val_map
        )
    )

    cmds = sorted(cmds, key=get_cmd_cuda_device)

    print("\n\n".join(cmds))

    if n_out_files > 0:
        start_file_id = 1
        curr_n_out_files = n_out_files // len(local_ranks)
        for _, cmds_cuda_group in groupby(cmds, key=get_cmd_cuda_device):
            cmds_cuda_group = list(cmds_cuda_group)
            write_cmds_device_group_to_files(
                cmds_cuda_group, curr_n_out_files, file_name_format,
                start_file_id
            )
            start_file_id += curr_n_out_files
    return 0


if __name__ == '__main__':
    sys.exit(main())
