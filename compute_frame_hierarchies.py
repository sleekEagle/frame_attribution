"""
compute_frame_hierarchies.py -- for every video in a dataset's eval split, compute its
adjacent-frame similarity hierarchy (frame_clustering.frame_hierarchy + hierarchy_groups) and
save the per-video results to one JSON-lines file per dataset in --out_dir.

The clusters are only meaningful for a model if they're built on the frames that model reads.
Models sample frames differently (motivation/e1_common.frame_indices_for):
    segment centres, 16 frames:  r3d, videomae
    segment centres, 8 frames:   trn, trn_official
    linspace, 16 frames:         vjepa2, mc3_18, r3d_18
So pass --models: each model gets its own file, built on exactly its frame indices, over its
own dataset (MODEL_SPECS[model]["dataset"]) unless --datasets is given.

    python compute_frame_hierarchies.py --models mc3_18 r3d_18 r3d      # UCF101 test split
    python compute_frame_hierarchies.py --models vjepa2 --datasets ssv2_sampled
    python compute_frame_hierarchies.py --datasets ssv2 ucf101_test     # no model: 16 segment centres

Output: <out_dir>/<dataset>_<model>_frame_hierarchy.jsonl with --models, else
<out_dir>/<dataset>_frame_hierarchy.jsonl (segment centres, --n_frames). One JSON object per line:
    {"path": str, "class": str, "model": str | null, "sampling": str, "n_frames": int,
     "frame_indices": [int, ...],
     "hierarchy": {"16": [[0],[1],...,[15]], "15": [[0,1],[2],...], ..., "1": [[0,...,15]]}}
frame_indices are the video frame numbers the clustering used (the model's frames).
hierarchy's keys are cluster counts k (finest k=n_frames down to coarsest k=1); each value is
that level's groups of positions 0..n_frames-1 in frame_indices, left-to-right, always
contiguous (see frame_clustering.hierarchy_groups).

Written incrementally, one line at a time with a flush after each video. Re-running resumes:
videos already in the output file are skipped (--overwrite starts from scratch).

Uses "ucf101_test" (dataloaders/ucf101_loader.get_ucf101_test_paths(), UCF101 split 1's
official testlist01.txt -- 3,783 held-out clips) rather than "ucf101" (which walks
CONST.UCF101_PATH's whole raw directory tree, train+test combined) so this is genuinely
eval-only, matching "ssv2" (CONST.SSV2_PATH already points at the s2s_test folder).
"""
import argparse
import json
from pathlib import Path

from torchcodec.decoders import VideoDecoder

from dataloaders.registry import get_dataloader
from frame_clustering import frame_hierarchy, hierarchy_groups
from models.video_utils import sample_segment_centers
from motivation.e1_common import MODEL_SPECS, frame_indices_for

DEFAULT_DATASETS = ["ssv2", "ucf101_test"]
DATASET_CHOICES = ["ssv2", "ssv2_sampled", "ucf101", "ucf101_test"]
MODEL_CHOICES = ["r3d", "videomae", "trn", "trn_official", "vjepa2", "mc3_18", "r3d_18"]


def _done_paths(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["path"])
    return done


def process_dataset(name: str, out_dir: Path, n_frames: int, model=None, overwrite=False) -> None:
    d_names, paths = get_dataloader(name)
    suffix = f"_{model}" if model else ""
    out_path = out_dir / f"{name}{suffix}_frame_hierarchy.jsonl"
    sampling = MODEL_SPECS[model].get("sampling", "centers") if model else "centers"
    done = set() if overwrite else _done_paths(out_path)
    n_total = len(paths)
    n_done = n_failed = 0
    if done:
        print(f"{out_path.name}: resuming, {len(done)} videos already done")

    with open(out_path, "w" if overwrite else "a", encoding="utf-8") as f:
        for idx, (cls, path) in enumerate(zip(d_names, paths)):
            if idx > 0:
                print(f"{out_path.name}: {idx / n_total * 100:.2f}% done ({n_failed} failed)", end="\r")
            if str(path) in done:
                continue
            try:
                n_video = len(VideoDecoder(str(path)))
                if model:
                    indices = frame_indices_for(model, n_video)
                else:
                    indices = sample_segment_centers(n_video, n_frames).tolist()
                Z, embeddings = frame_hierarchy(path, indices=indices)
                groups = hierarchy_groups(Z, len(embeddings))
            except Exception as e:
                print(f"\n[warn] skipping {path}: {e}")
                n_failed += 1
                continue
            record = {
                "path": str(path),
                "class": cls,
                "model": model,
                "sampling": sampling,
                "n_frames": len(embeddings),
                "frame_indices": [int(i) for i in indices],
                "hierarchy": groups,  # int keys -> json.dumps stringifies them ("16", "15", ...)
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_done += 1

    print(f"\n{out_path.name}: wrote {n_done} new videos ({len(done)} already done, "
          f"{n_failed} failed, {n_total} in dataset)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=None,
                        help="build each model's clusters on that model's own frame indices")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=None,
                        help="default: each model's own dataset, or ssv2 + ucf101_test without --models")
    parser.add_argument("--out_dir", default=r"D:\output\new_outs")
    parser.add_argument("--n_frames", type=int, default=16, help="only used without --models")
    parser.add_argument("--overwrite", action="store_true", help="start from scratch, don't resume")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.models:
        for model in args.models:
            for name in args.datasets or [MODEL_SPECS[model]["dataset"]]:
                process_dataset(name, out_dir, args.n_frames, model, args.overwrite)
    else:
        for name in args.datasets or DEFAULT_DATASETS:
            process_dataset(name, out_dir, args.n_frames, None, args.overwrite)
