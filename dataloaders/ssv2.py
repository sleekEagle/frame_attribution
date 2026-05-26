from pathlib import Path
import CONST

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