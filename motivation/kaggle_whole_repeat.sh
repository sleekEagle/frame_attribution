#!/usr/bin/env bash
# kaggle_whole_repeat.sh -- Kaggle jobs (2 x T4, one process per GPU with --shard 0/1):
#   * the whole-clip repetition check (every frame shown m times) for V-JEPA2
#     (motivation/whole_clip_repeat.py -> $OUT/vjepa2_whole_repeat.shard{0,1}.jsonl)
#   * the IG step check on every E1 model: do IG scores add up to the logit change, and does the E1
#     per-copy ratio change with more steps? (motivation/ig_completeness.py ->
#     $OUT/<model>_ig_completeness.shard{0,1}.jsonl, summaries in $OUT/ig_completeness_summary.txt)
# Logs go to $OUT/*_gpu<g>.log. Finished rows are skipped, so a stage can be rerun after a timeout.
#
# Set before running (in a notebook cell, e.g. %env SSV2_PATH=/kaggle/input/<dataset>/...):
#   SSV2_PATH    SSv2 videos, one sub-folder per class (V-JEPA2, TRN)
#   UCF101_PATH  UCF101 videos, one sub-folder per class (MC3-18, R3D-18, R3D-50)
#   REPO         this repository (default /kaggle/working/frame_attribution)
#   OUT          output folder (default /kaggle/working/e1); put the E1 files
#                <model>_conditions.jsonl here (results/e1/ is git-ignored)
#   R3D_CKPT     R3D-50 checkpoint save_200.pth, if models/r3d/ckpt/save_200.pth is missing
#   TRN_CKPT_DIR folder with TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar and
#                something_categories.txt, used if the download from the TRN authors' server fails
# MC3-18 and R3D-18 download their checkpoints from the Hugging Face Hub, V-JEPA2 too.
#
# Usage:  bash motivation/kaggle_whole_repeat.sh smoke     # V-JEPA2, 1 video per GPU, m = 1 and 3, IG + Grad-CAM
#         bash motivation/kaggle_whole_repeat.sh igcheck   # IG step check, 32/128/256 steps, on all five models
#                                                          # (MODELS="vjepa2 mc3_18" to run some of them)
#         bash motivation/kaggle_whole_repeat.sh m2        # V-JEPA2, m = 1, 2: Grad-CAM, LOO, occlusion, then Shapley
#         bash motivation/kaggle_whole_repeat.sh m3        # adds m = 3 for the same methods
#         bash motivation/kaggle_whole_repeat.sh summary   # both summaries again, from the files in $OUT
set -euo pipefail
STAGE=${1:?usage: kaggle_whole_repeat.sh smoke|igcheck|m2|m3|summary}
REPO=${REPO:-/kaggle/working/frame_attribution}
OUT=${OUT:-/kaggle/working/e1}
export SSV2_PATH=${SSV2_PATH:-} UCF101_PATH=${UCF101_PATH:-}
mkdir -p "$OUT"
cd "$REPO"
# IG is left out of the whole-clip check until igcheck shows how many steps V-JEPA2 needs: with
# 32 steps its scores did not add up to the logit change in the smoke test (sum / change -2 to 3)
FAST="gradcam loo_drop occlusion"
IG_MODELS=${MODELS:-"vjepa2 trn_official mc3_18 r3d_18 r3d"}
TRN_URL=http://relation.csail.mit.edu/models

cond_for() {   # cond_for <model>: the E1 conditions file, from $OUT or results/e1
  local f
  for f in "$OUT/$1_conditions.jsonl" "results/e1/$1_conditions.jsonl"; do
    [ -f "$f" ] && { echo "$f"; return 0; }
  done
  return 1
}

setup_model() {   # setup_model <model>: datasets and checkpoints; non-zero if the model can't run
  case $1 in
    vjepa2)
      [ -n "$SSV2_PATH" ] || { echo "SSV2_PATH not set"; return 1; } ;;
    trn_official)
      [ -n "$SSV2_PATH" ] || { echo "SSV2_PATH not set"; return 1; }
      # the Play Fair code (only its src/ is used; upstream is unchanged) and the original TRN checkpoint
      [ -d play-fair/src ] || git clone --depth 1 https://github.com/willprice/play-fair play-fair
      local d=play-fair/checkpoints/backbones f
      mkdir -p "$d"
      for f in TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar something_categories.txt; do
        [ -s "$d/$f" ] && continue
        if [ -n "${TRN_CKPT_DIR:-}" ] && [ -f "$TRN_CKPT_DIR/$f" ]; then cp "$TRN_CKPT_DIR/$f" "$d/$f"
        else curl -fsSL --max-time 600 -o "$d/$f" "$TRN_URL/$f" || { rm -f "$d/$f"; echo "could not get $f; set TRN_CKPT_DIR"; return 1; }
        fi
      done
      # the server answers dead links with an HTML page: the checkpoint must be a real file
      [ "$(stat -c %s "$d/TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar")" -gt 10000000 ] \
        || { echo "TRN checkpoint too small (an error page?); set TRN_CKPT_DIR"; return 1; } ;;
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

vjepa2_common() {   # vjepa2_common <out dir>: arguments shared by the V-JEPA2 whole-clip runs
  local c
  c=$(cond_for vjepa2) || { echo "vjepa2_conditions.jsonl not found in $OUT or results/e1" >&2; exit 1; }
  COMMON=(--model vjepa2 --conditions "$c" --out "$1" --fp16 --batch-size 16 --ig-batch 1 --num-shards 2)
}

run_both() {   # run_both <log name> <args...>: the same whole-clip command on GPU 0 and GPU 1, then wait
  local name=$1; shift
  for g in 0 1; do
    CUDA_VISIBLE_DEVICES=$g python motivation/whole_clip_repeat.py "${COMMON[@]}" --shard $g "$@" \
      > "$OUT/whole_repeat_${name}_gpu$g.log" 2>&1 &
  done
  wait
  tail -n 3 "$OUT"/whole_repeat_${name}_gpu*.log
}

ig_summaries() {
  local m
  for m in $IG_MODELS; do
    python motivation/ig_completeness.py --model "$m" --out "$OUT" --summary
    echo
  done | tee "$OUT/ig_completeness_summary.txt"
}

case $STAGE in
  smoke)
    OUT_SMOKE="$OUT/smoke"; mkdir -p "$OUT_SMOKE"
    setup_model vjepa2 || exit 1
    vjepa2_common "$OUT_SMOKE"
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
  igcheck)
    for model in $IG_MODELS; do
      echo "=== $model"
      if ! setup_model "$model"; then echo "skipping $model"; continue; fi
      cond=$(cond_for "$model") || { echo "${model}_conditions.jsonl not found in $OUT or results/e1; skipping"; continue; }
      limit=10; [ "$model" = vjepa2 ] && limit=4   # V-JEPA2 takes about 6 min per 32 steps on a long clip
      for g in 0 1; do
        CUDA_VISIBLE_DEVICES=$g python motivation/ig_completeness.py --model "$model" --conditions "$cond" \
          --out "$OUT" --limit $limit --steps 32 128 256 --shard $g --num-shards 2 \
          > "$OUT/ig_completeness_${model}_gpu$g.log" 2>&1 &
      done
      wait || echo "a $model process failed; see $OUT/ig_completeness_${model}_gpu*.log"
      tail -n 2 "$OUT"/ig_completeness_${model}_gpu*.log
    done
    ig_summaries
    ;;
  m2)
    setup_model vjepa2 || exit 1
    vjepa2_common "$OUT"
    run_both m2_fast    --limit 34 --ms 1 2 --methods $FAST
    run_both m2_shapley --limit 20 --ms 1 2 --methods shapley_drop
    ;;
  m3)
    setup_model vjepa2 || exit 1
    vjepa2_common "$OUT"
    run_both m3_fast    --limit 34 --ms 3 --methods $FAST
    run_both m3_shapley --limit 20 --ms 3 --methods shapley_drop
    ;;
  summary)
    python motivation/whole_clip_repeat.py --model vjepa2 --out "$OUT" --summary || true
    ig_summaries
    ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
