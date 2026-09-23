"""
compute_frame_hierarchies.py -- for every video in a dataset's eval split, compute its
adjacent-frame similarity hierarchy (frame_clustering.frame_hierarchy + hierarchy_groups) and
save the per-video results to one JSON-lines file per dataset in --out_dir.

    python compute_frame_hierarchies.py                     # ssv2 + ucf101, 16 frames/clip
    python compute_frame_hierarchies.py --datasets ssv2
    python compute_frame_hierarchies.py --out_dir D:\\output\\new_outs --n_frames 16

Output: <out_dir>/<dataset>_frame_hierarchy.jsonl, one JSON object per line:
    {"path": str, "class": str, "n_frames": int,
     "hierarchy": {"16": [[0],[1],...,[15]], "15": [[0,1],[2],...], ..., "1": [[0,...,15]]}}
hierarchy's keys are cluster counts k (finest k=n_frames down to coarsest k=1); each value is
that level's frame-index groups, left-to-right, always contiguous (see
frame_clustering.hierarchy_groups).

Written incrementally, one line at a time with a flush after each video, so a crash partway
through a large dataset (e.g. SSv2's eval split has ~27k clips) doesn't lose progress already
made -- resume by re-running (this script always overwrites the output file from scratch, it
does not skip videos already present in an existing file).

Uses "ucf101_test" (dataloaders/ucf101_loader.get_ucf101_test_paths(), UCF101 split 1's
official testlist01.txt -- 3,783 held-out clips) rather than "ucf101" (which walks
CONST.UCF101_PATH's whole raw directory tree, train+test combined) so this is genuinely
eval-only, matching "ssv2" (CONST.SSV2_PATH already points at the s2s_test folder).
"""
import argparse
import json
from pathlib import Path

from dataloaders.registry import get_dataloader
from frame_clustering import frame_hierarchy, hierarchy_groups

DEFAULT_DATASETS = ["ssv2", "ucf101_test"]


def process_dataset(name: str, out_dir: Path, n_frames: int) -> None:
    d_names, paths = get_dataloader(name)
    out_path = out_dir / f"{name}_frame_hierarchy.jsonl"
    n_total = len(paths)
    n_done = n_failed = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, (cls, path) in enumerate(zip(d_names, paths)):
            if idx > 0:
                print(f"{name}: {idx / n_total * 100:.2f}% done ({n_failed} failed)", end="\r")
            try:
                Z, embeddings = frame_hierarchy(path, n_frames=n_frames)
                groups = hierarchy_groups(Z, len(embeddings))
            except Exception as e:
                print(f"\n[warn] skipping {path}: {e}")
                n_failed += 1
                continue
            record = {
                "path": str(path),
                "class": cls,
                "n_frames": len(embeddings),
                "hierarchy": groups,  # int keys -> json.dumps stringifies them ("16", "15", ...)
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_done += 1

    print(f"\n{name}: wrote {n_done}/{n_total} videos to {out_path} ({n_failed} failed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                         choices=["ssv2", "ssv2_sampled", "ucf101", "ucf101_test"])
    parser.add_argument("--out_dir", default=r"D:\output\new_outs")
    parser.add_argument("--n_frames", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        process_dataset(name, out_dir, args.n_frames)
