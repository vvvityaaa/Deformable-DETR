from pathlib import Path
from .coco import CocoDetection, make_coco_transforms
from util.misc import get_local_rank, get_local_size


def build_custom(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'custom'
    PATHS = {
        "train": (root / "train2021", root / "annotations" / f'{mode}_train.json'),
        "val": (root / "val2021", root / "annotations" / f'{mode}_val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    return CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
        cache_mode=args.cache_mode,
        local_rank=get_local_rank(),
        local_size=get_local_size(),
    )