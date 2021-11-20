import pathlib
import shutil

def get_first_subdir_name(dir_path):
    return next(iter(dir_path.iterdir())).stem

def build_subdir_path(eval_dir):
    name = eval_dir.stem
    dataset, loss_token, solver_token, model = name.split('_')
    loss = loss_token.split('-')[1].replace(
        'contrastive', 'contr'
    ).replace(
        'triplet', 'tripl'
    )
    solver = solver_token.split('-')[1].replace(
        'featureemb', 'fNMS'
    ).replace(
        'original', 'orig'
    )
    model = model.replace('final', '0090000')
    tree_path = eval_dir.parent / dataset / loss / solver / model

    return tree_path

for eval_dir in pathlib.Path('./eval').iterdir():
    if '_' not in eval_dir.stem:
        continue

    subdir_name = get_first_subdir_name(eval_dir)
    src_eval_dir = eval_dir / subdir_name
    dst_eval_dir = build_subdir_path(eval_dir)
    dst_eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"processing: {src_eval_dir} --> {dst_eval_dir}")

    shutil.move(str(src_eval_dir), str(dst_eval_dir))
