"""
dataloaders/ucf101_loader.py -- path listing for UCF101's raw videos (CONST.UCF101_PATH: one
dir per class, one .avi per clip), mirroring dataloaders/ssv2.py's get_ssv2_paths() so both
datasets plug into dataloaders/registry.py's get_dataloader() the same way.
"""
from pathlib import Path
import CONST

VIDEO_EXTS = (".avi", ".mp4", ".webm")


def get_ucf101_paths():
    """Returns (d_names, paths): d_names[i] is the ground-truth class name (folder name) for
    paths[i], a video file. Feed paths[i] to a model's predict_from_path()."""
    root = Path(CONST.UCF101_PATH)
    d_names, paths = [], []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        vids = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        d_names.extend([cls_dir.name] * len(vids))
        paths.extend(vids)
    return d_names, paths


def get_ucf101_test_paths():
    """Returns (d_names, paths) restricted to UCF101 split 1's official test set
    (CONST.UCF101_SPLITS_PATH/testlist01.txt, 3,783 clips) -- the split R3D's checkpoint was
    fine-tuned and evaluated on (see CONST.UCF101_SPLITS_PATH's comment). Unlike
    get_ucf101_paths(), these are genuinely held-out clips, not train+test combined."""
    root = Path(CONST.UCF101_PATH)
    testlist = Path(CONST.UCF101_SPLITS_PATH) / "testlist01.txt"
    d_names, paths = [], []
    with open(testlist) as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            d_names.append(rel.split("/")[0])
            paths.append(root / rel)
    return d_names, paths
