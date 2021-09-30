import os
import sys
import shutil
import dataclasses
import subprocess
import itertools

from typing import Callable, Iterable, Iterator, Sequence, List, Tuple


@dataclasses.dataclass(frozen=True)
class CfgOptSpec:
    name: str
    vals: Sequence[str]

    def iter_name_val_pairs(self) -> Iterator[Tuple[str, str]]:
        yield from itertools.product((self.name,), self.vals)


CsvFilePathBuilderT = Callable[[Sequence[Tuple[str, str]]], str]


def _build_csv_file_path(cfg_opts: Sequence[Tuple[str, str]]) -> str:
    file_name = "eval"
    for opt_name, opt_val in cfg_opts:
        if "COS_SIM" in opt_name:
            file_name += "_cossim_" + opt_val.replace(".", "p")
        elif "DORMANT" in opt_name:
            file_name += "_ndormant_" + opt_val
        else:
            raise ValueError("unhandled option")
    file_name += ".csv"

    return file_name


def build_run_test_cmd(
    config_file_path,
    model_file_path,
    csv_file_path,
    cfg_opts
):
    run_test_args = [
        "python", "-m", "tools.test_net", "--config-file", config_file_path,
        "--model-file", model_file_path, "--set", "train",
        "--eval-csv-file", csv_file_path
    ]
    for cfg_opt in cfg_opts:
        run_test_args.extend(cfg_opt)

    return run_test_args


def iter_cmd_args(
    config_file_path: str,
    model_file_path: str,
    cmd_arg_specs: Iterable[CfgOptSpec],
    csv_file_path_builder: CsvFilePathBuilderT
) -> List[str]:
    cfg_name_val_iters = tuple(c.iter_name_val_pairs() for c in cmd_arg_specs)
    for cfg_opts in itertools.product(*cfg_name_val_iters):
        csv_file_path = csv_file_path_builder(cfg_opts)
        cmd = build_run_test_cmd(
            config_file_path, model_file_path, csv_file_path, cfg_opts
        )
        yield cmd


def main():
    # TODO Make these paths parametric using "click".
    inference_dump_dir_path = "./DLA-34-FPN_box_EMM"
    config_file_path = "./configs/dla/DLA_34_FPN_EMM_MOT17_test.yaml"
    model_file_path = "./demos/models/DLA-34-FPN_EMM_crowdhuman_mot17.pth"
    
    cmd_arg_specs = (
        CfgOptSpec(
            "MODEL.TRACK_HEAD.COS_SIM_THRESH",
            ("0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")
        ),
    )

    for cmd in iter_cmd_args(
        config_file_path, model_file_path, cmd_arg_specs,_build_csv_file_path
    ):
        if os.path.exists(inference_dump_dir_path):
            shutil.rmtree(inference_dump_dir_path)

        cmd_str = " ".join(cmd)
        print(f"Running command:\n{cmd_str}\n{'*' * 80}\n")

        subprocess.call(cmd, text=True, shell=True)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
