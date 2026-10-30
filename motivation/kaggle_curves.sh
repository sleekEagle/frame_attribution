#!/usr/bin/env bash
# kaggle_curves.sh -- deletion and insertion curves on Kaggle (2 x T4, one process per GPU with
# --shard 0/1): motivation/curves.py on the E2/E3 clips (MC3-18, R3D-18, R3D-50, V-JEPA2) and on the
# E4 clips (MC3-18, V-JEPA2), then motivation/curves_metrics.py for the summaries.
# Writes $OUT/<model>_curves_<source>.shard{0,1}.jsonl, $OUT/<model>_curves_<source>_summary.csv,
# $OUT/curves_summary.txt and logs $OUT/curves_<model>_<source>_gpu<g>.log. Finished clips are
# skipped, so a stage can be rerun after a timeout.
#
# Set before running (in a notebook cell, e.g. %env SSV2_PATH=/kaggle/input/<dataset>/...):
#   SSV2_PATH    SSv2 videos, one sub-folder per class (V-JEPA2)
#   UCF101_PATH  UCF101 videos, one sub-folder per class (MC3-18, R3D-18, R3D-50)
#   INPUTS       a folder with the git-ignored input files: <model>_e23.jsonl (4 models),
#                <model>_fr_practical.jsonl (mc3_18, vjepa2) and dino_tau.json, from results/e23,
#                results/fr and results/e1 (sub-folders are searched too)
#   REPO         this repository (default /kaggle/working/frame_attribution)
#   OUT          output folder (default /kaggle/working/curves)
#   R3D_CKPT     R3D-50 checkpoint save_200.pth, if models/r3d/ckpt/save_200.pth is missing
#   MODELS       models to run (default: all that the stage covers)
#
# Usage:  bash motivation/kaggle_curves.sh smoke     # 1 video per GPU: R3D-18 on E2/E3, V-JEPA2 on E4
#         bash motivation/kaggle_curves.sh e23       # E2/E3 clips, all four models
#         bash motivation/kaggle_curves.sh fr        # E4 clips, MC3-18 and V-JEPA2
#         bash motivation/kaggle_curves.sh summary   # the summaries again, from the files in $OUT
set -euo pipefail
STAGE=${1:?usage: kaggle_curves.sh smoke|e23|fr|summary}
REPO=${REPO:-/kaggle/working/frame_attribution}
OUT=${OUT:-/kaggle/working/curves}
export SSV2_PATH=${SSV2_PATH:-} UCF101_PATH=${UCF101_PATH:-}
mkdir -p "$OUT"
cd "$REPO"

copy_inputs() {   # the git-ignored inputs, from $INPUTS into results/
  [ -n "${INPUTS:-}" ] || { echo "INPUTS not set"; exit 1; }
  mkdir -p results/e23 results/fr results/e1
  local f
  while IFS= read -r f; do cp "$f" results/e23/; done < <(find "$INPUTS" -name '*_e23.jsonl')
  while IFS= read -r f; do cp "$f" results/fr/; done < <(find "$INPUTS" -name '*_fr_practical.jsonl')
  f=$(find "$INPUTS" -name dino_tau.json | head -n 1)
  [ -n "$f" ] && cp "$f" results/e1/ || { echo "dino_tau.json not found under $INPUTS"; exit 1; }
}

setup_model() {   # setup_model <model>: datasets and checkpoints; non-zero if the model can't run
  case $1 in
    vjepa2)
      [ -n "$SSV2_PATH" ] || { echo "SSV2_PATH not set"; return 1; } ;;
    r3d)
      [ -n "$UCF101_PATH" ] || { echo "UCF101_PATH not set"; return 1; }
      if [ ! -f models/r3d/ckpt/save_200.pth ]; then
        [ -n "${R3D_CKPT:-}" ] && [ -f "$R3D_CKPT" ] || { echo "models/r3d/ckpt/save_200.pth missing; set R3D_CKPT"; return 1; }
        mkdir -p models/r3d/ckpt && cp "$R3D_CKPT" models/r3d/ckpt/save_200.pth
      fi ;;
    mc3_18|r3d_18)
      [ -n "$UCF101_PATH" ] || { echo "UCF101_PATH not set"; return 1; } ;;
  esac
}

run_both() {   # run_both <model> <source> <out dir> [extra args...]: one shard per GPU, then wait
  local model=$1 source=$2 out=$3; shift 3
  local bs=32; [ "$model" = vjepa2 ] && bs=16
  for g in 0 1; do
    CUDA_VISIBLE_DEVICES=$g python motivation/curves.py --model "$model" --source "$source" \
      --out "$out" --fp16 --batch-size $bs --shard $g --num-shards 2 "$@" \
      > "$out/curves_${model}_${source}_gpu$g.log" 2>&1 &
  done
  wait || echo "a $model $source process failed; see $out/curves_${model}_${source}_gpu*.log"
  tail -n 2 "$out"/curves_${model}_${source}_gpu*.log
}

run_stage() {   # run_stage <source> <models...>
  local source=$1 model; shift
  for model in "$@"; do
    echo "=== $model $source"
    setup_model "$model" || { echo "skipping $model"; continue; }
    run_both "$model" "$source" "$OUT"
  done
}

summaries() {
  local f source model
  for f in "$OUT"/*_curves_*.shard*.jsonl; do
    [ -e "$f" ] || continue
    f=${f##*/}; f=${f%.shard*}
    echo "${f%_curves_*} ${f##*_curves_}"
  done | sort -u | while read -r model source; do
    python motivation/curves_metrics.py --model "$model" --source "$source" --out "$OUT"
    echo
  done | tee "$OUT/curves_summary.txt"
}

case $STAGE in
  smoke)
    copy_inputs
    mkdir -p "$OUT/smoke"
    setup_model r3d_18 && run_both r3d_18 e23 "$OUT/smoke" --limit 2
    setup_model vjepa2 && run_both vjepa2 fr "$OUT/smoke" --limit 2
    python - "$OUT/smoke" <<'EOF'
import json, glob, sys
for p in sorted(glob.glob(sys.argv[1] + "/*_curves_*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        print(f"{p.split('/')[-1]}: {r['clip']}, n={r['n']}, {r['n_evals']} evals, "
              f"methods={sorted(r['curves'])}")
EOF
    grep -h "s, " "$OUT"/smoke/curves_*_gpu*.log | tail -n 4 || true
    ;;
  e23)
    copy_inputs
    run_stage e23 ${MODELS:-r3d_18 mc3_18 r3d vjepa2}
    summaries
    ;;
  fr)
    copy_inputs
    run_stage fr ${MODELS:-mc3_18 vjepa2}
    summaries
    ;;
  summary)
    summaries
    ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
