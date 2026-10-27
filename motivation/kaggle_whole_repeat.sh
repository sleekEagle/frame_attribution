#!/usr/bin/env bash
# kaggle_whole_repeat.sh -- the whole-clip repetition check (every frame shown m times) for V-JEPA2
# on Kaggle, 2 x T4. One process per GPU (--shard 0/1); results go to $OUT as
# vjepa2_whole_repeat.shard0.jsonl / .shard1.jsonl, logs to $OUT/whole_repeat_<stage>_gpu<g>.log.
# Already finished (video, m, method) rows are skipped, so a stage can be rerun after a timeout.
#
# Set before running (e.g. in a notebook cell: %env SSV2_PATH=/kaggle/input/<ssv2 dataset>/...):
#   SSV2_PATH  folder with one sub-folder per SSv2 class, as for the earlier V-JEPA2 Kaggle jobs
#   REPO       this repository (default /kaggle/working/frame_attribution)
#   OUT        output folder (default /kaggle/working/e1)
#   COND       E1 conditions file (default $OUT/vjepa2_conditions.jsonl, else results/e1/...)
#
# Usage:  bash motivation/kaggle_whole_repeat.sh smoke     # 1 video per GPU, m = 1 and 3, IG + Grad-CAM:
#                                                          # checks GPU memory at 48 frames and the time per method
#         bash motivation/kaggle_whole_repeat.sh m2        # m = 1, 2: IG, Grad-CAM, LOO, occlusion, then Shapley
#         bash motivation/kaggle_whole_repeat.sh m3        # adds m = 3 for the same methods
#         bash motivation/kaggle_whole_repeat.sh summary
set -euo pipefail
STAGE=${1:?usage: kaggle_whole_repeat.sh smoke|m2|m3|summary}
REPO=${REPO:-/kaggle/working/frame_attribution}
OUT=${OUT:-/kaggle/working/e1}
: "${SSV2_PATH:?set SSV2_PATH to the SSv2 videos (one folder per class)}"
export SSV2_PATH
mkdir -p "$OUT"
cd "$REPO"
# 34 videos; their paths are remapped under SSV2_PATH. results/e1/ is git-ignored, so upload this
# file with the code, or copy it to $OUT as for the earlier E1 Kaggle jobs.
COND=${COND:-$OUT/vjepa2_conditions.jsonl}
[ -f "$COND" ] || COND=results/e1/vjepa2_conditions.jsonl
[ -f "$COND" ] || { echo "vjepa2_conditions.jsonl not found in $OUT or results/e1; set COND"; exit 1; }
echo "conditions: $COND"
COMMON=(--model vjepa2 --conditions "$COND" --out "$OUT" --fp16 --batch-size 16 --ig-batch 1 --num-shards 2)
FAST="ig gradcam loo_drop occlusion"

run_both() {   # run_both <log name> <args...>: the same command on GPU 0 and GPU 1, then wait for both
  local name=$1; shift
  for g in 0 1; do
    CUDA_VISIBLE_DEVICES=$g python motivation/whole_clip_repeat.py "${COMMON[@]}" --shard $g "$@" \
      > "$OUT/whole_repeat_${name}_gpu$g.log" 2>&1 &
  done
  wait
  tail -n 3 "$OUT"/whole_repeat_${name}_gpu*.log
}

case $STAGE in
  smoke)
    OUT_SMOKE="$OUT/smoke"; mkdir -p "$OUT_SMOKE"
    COMMON=(--model vjepa2 --conditions "$COND" --out "$OUT_SMOKE" --fp16 --batch-size 16 --ig-batch 1 --num-shards 2)
    run_both smoke --limit 2 --ms 1 3 --methods gradcam ig
    python motivation/whole_clip_repeat.py --model vjepa2 --out "$OUT_SMOKE" --summary
    python - "$OUT_SMOKE" <<'EOF'
import json, glob, sys
for p in sorted(glob.glob(sys.argv[1] + "/vjepa2_whole_repeat.shard*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        print(f"m={r['m']} {r['method']:<8} {r['secs']:8.1f} s  same_pred={r['same_pred']}")
EOF
    ;;
  m2)
    run_both m2_fast    --limit 34 --ms 1 2 --methods $FAST
    run_both m2_shapley --limit 20 --ms 1 2 --methods shapley_drop
    ;;
  m3)
    run_both m3_fast    --limit 34 --ms 3 --methods $FAST
    run_both m3_shapley --limit 20 --ms 3 --methods shapley_drop
    ;;
  summary)
    python motivation/whole_clip_repeat.py --model vjepa2 --out "$OUT" --summary
    ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
