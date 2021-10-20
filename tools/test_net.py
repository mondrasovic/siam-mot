"""
Basic testing script for PyTorch
Only support single-gpu now
"""

import argparse
import os

import torch
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from siammot.configs.defaults import cfg
from siammot.data.adapters.handler.data_filtering import build_data_filter_fn
from siammot.data.adapters.utils.data_utils import (
    load_dataset_anno,
    load_public_detection,
)
from siammot.engine.inferencer import DatasetInference
from siammot.modelling.rcnn import build_siammot
from siammot.utils.get_model_name import get_model_name


try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

parser = argparse.ArgumentParser(
    description="PyTorch Video Object Detection Inference"
)
parser.add_argument(
    "--config-file", default="", metavar="FILE", help="path to config file",
    type=str
)
parser.add_argument(
    "--output-dir", default="", help="path to output folder", type=str
)
parser.add_argument(
    "--model-file", default=None, metavar="FILE", help="path to model file",
    type=str
)
parser.add_argument("--test-dataset", default="MOT17_DPM", type=str)
parser.add_argument("--set", default="test", type=str)
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--num-gpus", default=1, type=int)
parser.add_argument("--eval-csv-file", default=None, type=str)
parser.add_argument(
    'opts', help="overwriting the training config from commandline",
    default=None, nargs=argparse.REMAINDER
)

def test(cfg, args, output_dir):
    torch.cuda.empty_cache()
    
    # Construct model graph
    model = build_siammot(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    
    # Load model params
    model_file = args.model_file
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=model_file)
    if os.path.isfile(model_file):
        checkpointer.load(model_file)
    elif os.path.isdir(model_file):
        checkpointer.load(use_latest=True)
    else:
        raise KeyError("No checkpoint is found")
    
    # Load testing dataset
    dataset_key = args.test_dataset
    dataset, _ = load_dataset_anno(cfg, dataset_key, args.set)
    dataset = sorted(dataset)
    
    # do inference on dataset
    data_filter_fn = build_data_filter_fn(dataset_key, dataset=dataset)
    
    # load public detection
    public_detection = None
    if cfg.INFERENCE.USE_GIVEN_DETECTIONS:
        public_detection = load_public_detection(cfg, dataset_key)
    
    dataset_inference = DatasetInference(
        cfg, model, dataset, output_dir, data_filter_fn, public_detection,
        motsummary_csv_file_name=args.eval_csv_file
    )
    dataset_inference()


def main():
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    model_name = get_model_name(cfg)
    output_dir = os.path.join(args.output_dir, model_name)
    if not os.path.exists(output_dir):
        mkdir(output_dir)
    
    test(cfg, args, output_dir)


if __name__ == "__main__":
    main()
