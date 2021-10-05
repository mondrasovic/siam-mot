import json
import shutil
import pathlib
import sys
import click


def anno_extract_entities(anno, sample_name, entity_predicate):
    sample_data = anno['samples'][sample_name]
    entities_orig = sample_data['entities']
    entities_new = []
    unique_frame_idxs = set()

    for entity in filter(entity_predicate, entities_orig):
        entities_new.append(entity)
        frame_idx = entity['blob']['frame_idx']
        unique_frame_idxs.add(frame_idx)
    
    sample_metadata = sample_data['metadata']
    sample_metadata['number_of_frames'] = len(unique_frame_idxs)

    anno_new = {
        'metadata': anno['metadata'],
        'samples': {
            sample_name: {
                'entities': entities_new,
                'metadata': sample_metadata,
            }
        }
    }
    
    return anno_new


def create_splits_dict(subset_name, sample_name):
    other_subset_name = 'test' if subset_name == 'train' else 'train'

    splits_dict = {
        subset_name: [sample_name],
        other_subset_name: [],
    }

    return splits_dict


def recreate_splits_file(splits_file_path, subset_name, sample_name):
    with open(splits_file_path, 'wt') as splits_fp:
        splits_dict = create_splits_dict(subset_name, sample_name)
        json.dump(splits_dict, splits_fp, indent=2)


def recreate_anno_file(
    anno_src_file_path,
    anno_dst_file_path,
    sample_name,
    entity_predicate
):
    with open(anno_src_file_path) as anno_fp:
            anno_src = json.load(anno_fp)
        
    with open(anno_dst_file_path, 'wt') as anno_fp:
        anno_dst = anno_extract_entities(
            anno_src, sample_name, entity_predicate
        )
        json.dump(anno_dst, anno_fp, indent=2)


def recreate_data_subset_hierarchy(
    src_dataset_dir_path,
    dst_dataset_dir_path,
    subset_name,
    sample_name,
    entity_predicate,
    file_predicate
):
    src_dataset_dir = pathlib.Path(src_dataset_dir_path)
    dst_dataset_dir = pathlib.Path(dst_dataset_dir_path)

    if dst_dataset_dir.exists():
       shutil.rmtree(str(dst_dataset_dir))
    
    src_anno_dir = src_dataset_dir / 'annotation'
    dst_anno_dir = dst_dataset_dir / 'annotation'
    dst_anno_dir.mkdir(exist_ok=True, parents=True)

    splits_file_path = str(dst_anno_dir / 'splits.json')
    recreate_splits_file(splits_file_path, subset_name, sample_name)

    anno_src_file_path = str(src_anno_dir / 'anno.json')
    anno_dst_file_path = str(dst_anno_dir / 'anno.json')
    recreate_anno_file(
        anno_src_file_path, anno_dst_file_path, sample_name, entity_predicate
    )

    subset_dir_name = 'Insight-MVT_Annotation_' + subset_name.capitalize()
    rel_dir = pathlib.Path('raw_data') / subset_dir_name / sample_name
    src_imgs_dir = src_dataset_dir / rel_dir
    dst_imgs_dir = dst_dataset_dir / rel_dir
    dst_imgs_dir.mkdir(exist_ok=True, parents=True)

    for img_file in filter(file_predicate, src_imgs_dir.iterdir()):
        src_img_file_path = str(img_file)
        dst_img_file_path = str(dst_imgs_dir / img_file.name)
        shutil.copy2(src_img_file_path, dst_img_file_path)


def build_frame_idx_in_range_predicate(frame_idx_min, frame_idx_max):
    def _predicate(entity):
        return frame_idx_min <= entity['blob']['frame_idx'] <= frame_idx_max
    return _predicate


def build_img_file_in_range_predicate(frame_no_min, frame_no_max):
    def _predicate(img_file):
        return frame_no_min <= int(img_file.stem[3:]) <= frame_no_max
    return _predicate


@click.command()
@click.argument('src_dir_path', type=click.Path())
@click.argument('dst_dir_path', type=click.Path())
@click.argument('sample', type=str)
@click.option(
    '-s', '--subset', type=click.Choice(['train', 'test']),
    multiple=False, show_default=True, help="Data subset."
)
@click.option(
    '-b', '--begin-frame', type=click.IntRange(min=1), default=1,
    show_default=True, help="Minimum frame no."
)
@click.option(
    '-e', '--end-frame', type=click.IntRange(min=1), default=None,
    show_default=True, help="Maximum frame no."
)
def main(
    src_dir_path,
    dst_dir_path,
    sample,
    subset,
    begin_frame,
    end_frame
):
    if end_frame is None:
        end_frame = sys.size
    
    if end_frame < begin_frame:
        begin_frame, end_frame = end_frame, begin_frame
    
    entity_frame_range_predicate = build_frame_idx_in_range_predicate(
        begin_frame - 1, end_frame - 1
    )
    img_frame_range_predicate = build_img_file_in_range_predicate(
        begin_frame, end_frame
    )

    recreate_data_subset_hierarchy(
        src_dir_path, dst_dir_path, subset, sample,
        entity_frame_range_predicate, img_frame_range_predicate
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
