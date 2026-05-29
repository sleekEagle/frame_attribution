from pathlib import Path
import CONST
import os

def get_ssv2_paths():
    path = Path(CONST.SSV2_PATH)
    dirs = [p.name for p in path.iterdir() if p.is_dir()]
    dirs = [p for p in path.iterdir() if p.is_dir()]
    n_files = len([p for p in path.rglob("*") if p.is_file()])

    d_names = []
    paths = []
    for dir in dirs:
        d_name = dir.name
        files = [p for p in dir.iterdir() if p.is_file()]
        d_names.extend([d_name]*len(files))
        paths.extend(files)

    return d_names, paths
from pathlib import Path
def get_sampled_paths():
    path = Path(CONST.SSV2_PATH)
    cls_list, path_list = [], []
    with open('dataloaders/ssv2_paths.txt', 'r') as f:
        for line in f:
            full = Path(line.strip())
            cls = os.path.dirname(full)
            full = os.path.join(path, full)
            cls_list.append(cls)
            path_list.append(full)
    return cls_list, path_list





