import sys
import pathlib

import click
import torch


def load_model_state_dict(src_model_file_path):
    state_dict = torch.load(src_model_file_path)
    
    keys_to_remove = set(state_dict.keys()) - {'model'}
    for key in keys_to_remove:
        del state_dict[key]
    
    return state_dict


def save_model_state_dict(model_state_dict, dst_model_file_path):
    torch.save(model_state_dict, dst_model_file_path)


def create_last_checkpoint_file(dst_model_dir, last_model_file_path):
    last_checkpoint_file = dst_model_dir / 'last_checkpoint'
    last_checkpoint_file.write_text(last_model_file_path)


@click.command()
@click.argument('src_model_file_path', type=click.Path(exists=True))
@click.argument('dst_model_dir_path', type=click.Path())
@click.option(
    '-m', '--model-file-name', default='model_pretrained.pth',
    show_default=True, help="Model checkpoint file name."
)
def main(src_model_file_path, dst_model_dir_path, model_file_name):
    dst_model_dir = pathlib.Path(dst_model_dir_path)
    model_state_dict = load_model_state_dict(src_model_file_path)
    dst_model_file_path = str(dst_model_dir / model_file_name)
    save_model_state_dict(model_state_dict, dst_model_file_path)
    create_last_checkpoint_file(dst_model_dir, dst_model_file_path)

    return 0


if __name__ == '__main__':
    sys.exit(main())