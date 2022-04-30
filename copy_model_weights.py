import sys

import click
import torch


def load_model_weights(model_or_checkpoint_file_path):
    dict_data = torch.load(model_or_checkpoint_file_path)
    model_data = dict_data['model'] if 'model' in dict_data else dict_data
    return model_data


def replace_common_weights(src_model, dst_model):
    for src_layer_name, src_weights in src_model.items():
        if src_layer_name in dst_model:
            dst_model[src_layer_name] = src_weights


@click.command()
@click.argument('src_model_file_path', type=click.Path(exists=True))
@click.argument('dst_model_file_path', type=click.Path(exists=True))
@click.argument('out_model_file_path', type=click.Path())
def main(src_model_file_path, dst_model_file_path, out_model_file_path):
    src_model = load_model_weights(src_model_file_path)
    dst_model = load_model_weights(dst_model_file_path)

    replace_common_weights(src_model, dst_model)
    torch.save(dst_model, out_model_file_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())
