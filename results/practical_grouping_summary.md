# Practical groupings: removing dilution without knowing which frames are redundant

This summary covers the follow-up to `results/group_attribution_summary.md`. There, group values removed dilution, but only with groupings that used knowledge we don't have for a new video: the true copies (oracle), or DINOv2 cut to the oracle's number of groups. Here, every grouping is chosen in advance, without that knowledge. The question is also whether similarity-based groups do better than simple fixed windows.

## In short

**Grouping frames by DINOv2 similarity, with a fixed average-linkage threshold (`dinoavg_thr`), removes dilution in all 7 tests.** That's E1 on five models, and the frame-rate test on two. The duplicated or more densely sampled content keeps 0.96–1.07 of its earlier credit, while frame-level Shapley gives it 0.16–0.62.

**Fixed windows don't remove it.** With the same number of groups as DINOv2, windows give 0.35–0.66 of the earlier credit (0.83–0.90 on MC3-18). Windows of a fixed width give 0.26–0.65 on the models whose clips get longer. Both are about as diluted as frame-level Shapley. Even with the same number of groups as DINOv2, windows split repeated content across several windows. On R3D-18, 8 DINOv2 groups give 1.07 and 8 windows give 0.58.

**The choice of clustering matters:**
- **A fixed number of DINOv2 groups** works with copies, but not fully when a segment is sampled more densely. On V-JEPA2: 0.79–0.89.
- **A threshold with Ward linkage** fails on denser sampling (V-JEPA2: 0.63). Ward's merge height grows with cluster size, so a longer run of similar frames is split into more groups.
- **Average linkage doesn't have this size effect,** and is the variant to recommend.

## 1. How the test works

### 1.1 Groupings

All settings are fixed per model before the test, and never use the true copies:

| name | groups |
|---|---|
| `dino_k<K>` | the DINOv2 tree (neighbouring frames only, Ward linkage) cut into a fixed K groups |
| `dino_thr` | the same Ward tree, cut at a fixed merge height τ |
| `dinoavg_thr` | the same with an average-linkage tree, cut at its own τ |
| `win_k<K>` | K equal windows of neighbouring frames |
| `win_w<W>` | windows of W frames; the number of windows grows with the clip length |

K = 4 and 8 on the 16-frame models, and 2 and 4 on TRN (8 frames). The threshold groupings keep between 2 and 8 groups. Exact Shapley over more than 8 groups was too expensive on the larger models.

**How τ was chosen.** For each model and linkage, τ is set so that the median clip has half as many groups as frames: 8 for 16 frames, 4 for TRN. It's calibrated on videos that aren't used in the tests: the E2/E3 videos for the four 16-frame models, and 150 other SSv2 videos for TRN (`motivation/calibrate_tau.py`). Then it's frozen.

| model | τ, Ward | τ, average |
|---|---|---|
| R3D-50 | 0.404 | 0.367 |
| R3D-18 | 0.424 | 0.397 |
| MC3-18 | 0.420 | 0.389 |
| V-JEPA2 | 0.413 | 0.374 |
| TRN | 0.484 | 0.453 |

### 1.2 Readout: broadcast scores

Each grouping gives group Shapley values, computed exactly with each group as one player, and with the same class, removal and prior as in E1. Every frame then shows **the value of its group, not divided**. That's what a user would see if a group explanation were drawn over the frames. These broadcast scores are read exactly like a frame-level method's scores:
- **E1:** the per-copy score of the target, relative to the reference clip; β; inversions validated by the model's own reliance; top-1 loss.
- **Frame-rate test:** the average score of the denser segment's frames, relative to the base clip; β; inversions; top-1 loss.

The yardstick is always the model's own reliance (the I_t ratio, or the segment's I ratio), never group Shapley.

A tie for first counts as still top, because frames in one group share a score. The "total over copies" readout isn't meaningful for broadcast scores, so it isn't used.

### 1.3 Data

- **E1:** exact copies; the same clips as the step-2 group follow-up. 60 videos for R3D-50, R3D-18 and MC3-18; all 245 for TRN; all 34 for V-JEPA2.
- **Frame-rate test:** all 48 MC3-18 videos and 29 V-JEPA2 videos.
- **Frame-level Shapley** is shown on the same clips, as the baseline.

## 2. Results: E1 (duplicated frames)

Score per copy at the highest number of copies, relative to the reference. β in brackets.

| | TRN (×2) | R3D-18 (×4) | R3D-50 (×4) | MC3-18 (×9) | V-JEPA2 (×9) |
|---|---|---|---|---|---|
| model's reliance (I_t ratio) | 1.32 | 1.27 | 1.01 | 2.88 | 0.87 |
| frame Shapley | 0.50 (−0.94) | 0.54 (−0.45) | 0.62 (−0.36) | 0.49 (−0.30) | 0.16 (−0.71) |
| **`dinoavg_thr`** | **1.07 (+0.12)** | **1.04 (+0.05)** | **1.05 (+0.04)** | **1.05 (+0.03)** | **0.97 (−0.01)** |
| `dino_thr` | 1.13 (+0.19) | 1.23 (+0.13) | 1.19 (+0.09) | 0.96 (−0.02) | 0.73 (−0.14) |
| `dino_k`, larger K | 1.12 (+0.17) | 1.07 (+0.05) | 1.09 (+0.06) | 0.97 (0.00) | 0.72 (−0.10) |
| `dino_k`, smaller K | 1.00 (0.00) | 0.94 (−0.03) | 0.98 (−0.03) | 1.02 (+0.02) | – |
| `win_k`, larger K | 0.58 (−0.78) | 0.58 (−0.39) | 0.66 (−0.29) | 0.83 (−0.07) | 0.35 (−0.31) |
| `win_k`, smaller K | 0.74 (−0.39) | 0.70 (−0.20) | 0.81 (−0.15) | 0.93 (−0.02) | – |
| `win_w`, fixed width | 0.58 (−0.78) | same as `win_k4` | same as `win_k4` | 0.65 (−0.18) | 0.26 (−0.49) |

"Larger K" is K = 4 on TRN, K = 8 on R3D-18, R3D-50 and MC3-18, and K = 4 on V-JEPA2, which ran only K = 4 to save compute. TRN, R3D-18 and R3D-50 keep a fixed clip length, so their fixed-width windows are the same as `win_k4`.

The 95% ranges of `dinoavg_thr`'s β all lie between −0.05 and +0.19. Those of the windows with the larger K all lie below −0.05. In the control clips, every grouping stays close to 1 (0.91–1.12), so the changes in the target clips are caused by duplication. The exception is fixed-width windows on the insertion models (MC3-18 0.70, V-JEPA2 0.86): there the control clip is longer too, so it also gets more windows.

**Inversions** (validated by the model's own reliance), target clips at the highest number of copies:

| | TRN | R3D-18 | R3D-50 | MC3-18 | V-JEPA2 |
|---|---|---|---|---|---|
| frame Shapley | 0.36 | 0.19 | 0.13 | 0.02 | 0.19 |
| `dinoavg_thr` | 0.01 | 0.04 | 0.01 | 0.08 | 0.08 |
| `win_k`, larger K | 0.33 | 0.33 | 0.25 | 0.06 | 0.26 |

**Top-1 loss** on the two models with enough cases:
- **R3D-18:** frame Shapley 0.72, `dinoavg_thr` 0.03, `win_k8` 0.46.
- **TRN:** 0.61, 0.04 and 0.46 (`win_k4`).

On R3D-50 and MC3-18, top-1 loss is high in the control clips too, and on V-JEPA2 there are too few cases.

**Where the copies end up:**
- **Every DINOv2 grouping** puts all copies of the target in one group, in all five models.
- **Windows** spread them over 1.5–4.5 windows on average: the more windows, the stronger the dilution.

## 3. Results: the frame-rate test (denser real frames)

The target segment sampled 3 times as densely. Average score per frame relative to the base clip; β in brackets; and inversions in the base clip → at d = 3.

| | MC3-18 (48 videos) | V-JEPA2 (29 videos) | V-JEPA2 inversions |
|---|---|---|---|
| model's reliance on the segment | 0.90 | 0.94 | – |
| frame Shapley | 0.62 (−0.44) | 0.49 (−0.50) | 0.26 → 0.41 |
| **`dinoavg_thr`** | **0.98 (−0.02)** | **1.04 (+0.04)** | **0.21 → 0.23** |
| `dino_thr` | 0.77 (−0.20) | 0.63 (−0.38) | 0.19 → 0.42 |
| `dino_k4` | 0.98 (−0.02) | 0.89 (−0.12) | 0.36 → 0.40 |
| `dino_k8` | 0.94 (−0.06) | 0.79 (−0.17) | 0.21 → 0.42 |
| `win_k4` | 0.90 (−0.11) | 0.59 (−0.56) | 0.00 → 0.42 |
| `win_k8` | 0.90 (−0.10) | 0.59 (−0.43) | 0.06 → 0.40 |
| `win_w4` | 0.61 (−0.44) | 0.37 (−0.85) | 0.00 → 0.38 |

When the control segment is densified instead, the target's score stays near 1 for every grouping on V-JEPA2. The exception is `win_k8` (+0.29): the clip gets longer, so its windows change. On MC3-18, frame Shapley's and `win_w4`'s scores also fall in the control (β −0.20 and −0.28), because the clip gets longer.

**What this shows:**
- **Only `dinoavg_thr` keeps the denser segment's credit and its ranking on V-JEPA2.** Its inversions don't change (0.21 → 0.23), while every other method's roughly double.
- **Fixed windows dilute.** `win_w4` dilutes even more than frame Shapley on V-JEPA2 (0.37), because a denser segment covers more windows.
- **A fixed number of DINOv2 groups dilutes partly.** With K fixed, the extra frames have to be split across groups.
- **The Ward threshold fails** (0.63). A longer run of similar frames merges later under Ward, so a fixed threshold splits it.

On V-JEPA2, the windows' inversions are 0 at the base clip by construction, because `win_k4` there is exactly the 4 test segments.

## 4. Resolution, uncertainty and cost

These three analyses use the existing files only (`motivation/practical_extras.py`).

### 4.1 How coarse the groupings are

Number of groups and frames per group on the clips a user would see: the E1 reference clips, and the frame-rate test base clips.

| | TRN (8 frames) | R3D-18 | R3D-50 | MC3-18 | V-JEPA2 |
|---|---|---|---|---|---|
| `dinoavg_thr`: groups, median [10–90%] | 3 [2–4] | 6 [2–8] | 8 [3–8] | 8 [2–8] | 8 [2–8] |
| `dinoavg_thr`: frames per group | 2.5 | 3.0 | 2.5 | 2.5 | 2.6 |
| `dino_k8` and `win_k8`: frames per group | 2.0 (K = 4) | 2.0 | 2.0 | 2.0 | – |

**The remedy coarsens the explanation by a factor of about 2.5,** from single frames to about 2.5 frames per group. Windows with 2 frames per group still dilute (section 2). So the coarser resolution alone doesn't explain why `dinoavg_thr` works; what matters is that it puts similar frames together. In 10% of clips, though, it merges a clip into 2–3 groups, which is much coarser. The MC3-18 frame-rate run allowed up to 12 groups, so its median there is 9.

### 4.2 Ranking readouts with 95% confidence intervals

The intervals come from resampling videos (2,000 resamples). Validated inversions in the E1 target clips at the highest number of copies:

| | frame Shapley | `dinoavg_thr` | windows, same K |
|---|---|---|---|
| TRN | 0.36 [0.31, 0.42] | **0.01 [0.00, 0.03]** | 0.33 [0.28, 0.39] |
| R3D-18 | 0.19 [0.14, 0.25] | **0.04 [0.01, 0.08]** | 0.33 [0.26, 0.40] |
| R3D-50 | 0.13 [0.07, 0.20] | **0.01 [0.00, 0.03]** | 0.25 [0.17, 0.34] |
| V-JEPA2 | 0.19 [0.13, 0.26] | 0.08 [0.01, 0.18] | 0.26 [0.14, 0.40] |
| MC3-18 | 0.02 [0.01, 0.03] | 0.08 [0.03, 0.13] | 0.06 [0.03, 0.09] |

Top-1 loss:

| | frame Shapley | `dinoavg_thr` | windows, same K |
|---|---|---|---|
| TRN, E1 | 0.61 [0.52, 0.70] | **0.04 [0.01, 0.07]** | 0.46 [0.37, 0.55] |
| R3D-18, E1 | 0.72 [0.52, 0.88] | **0.03 [0.00, 0.09]** | 0.46 [0.25, 0.67] |
| V-JEPA2, frame-rate test (d = 3) | 0.71 [0.43, 0.93], 14 cases | **0.00**, 13 cases | 0.71 [0.54, 0.89] |

In the V-JEPA2 frame-rate test at d = 3, the inversions are, target segment densified (control segment densified):
- frame Shapley: 0.41 [0.28, 0.53] (0.12);
- `dinoavg_thr`: 0.23 [0.11, 0.37] (0.26);
- windows: 0.38–0.42 (0.02–0.15).

**What the intervals support:**
- **TRN, R3D-18 and R3D-50:** `dinoavg_thr` gives clearly fewer inversions than both frame Shapley and windows; the intervals don't overlap. On TRN and R3D-18 it also gives clearly less top-1 loss.
- **V-JEPA2 frame-rate test:** the intervals overlap. But `dinoavg_thr`'s inversions are the same as in its control, while every other method's rise 3–20-fold above their control. So the supported claim is that densifying doesn't increase its ranking errors, not that it has fewer errors in absolute terms.
- **V-JEPA2 E1:** fewer inversions (0.08 against 0.19), but the intervals overlap. It's suggestive only.
- **MC3-18:** not supported. Inversions are rare for every method, and frame Shapley has the fewest. MC3-18's parts matter about equally (see E3), so there's little ranking to get wrong.
- **R3D-50 top-1 loss** is as high in the control clips as in the target clips (frame Shapley 0.61 and 0.61), so it isn't used.

### 4.3 Cost

Model evaluations per clip (medians):

| | frame Shapley (sampled orderings) | `dinoavg_thr` (exact) | `dino_k8` and `win_k8` (exact) |
|---|---|---|---|
| TRN | 510 | 14 | 30 (K = 4) |
| R3D-18, R3D-50 | 1,662 | 30–62 | 510 |
| MC3-18 | 1,115 | 255 | 255 |
| V-JEPA2 | 571 | 255 (frame-rate test: 127) | 255 |

**Group Shapley over the `dinoavg_thr` groups needs about 2–35 times fewer model evaluations than frame-level Shapley,** and its values are exact, while frame-level Shapley's are sampled estimates (64 or 32 orderings). The R3D and TRN counts include a factor of 2, because their removal averages two ways of filling a gap. Embedding the frames with DINOv2 costs one pass over 16 frames, which is small next to the video model's evaluations.

## 5. Conclusions

1. **A practical grouping can remove dilution.** DINOv2 similarity with a fixed average-linkage threshold, calibrated once per model on other videos, keeps repeated content's credit at 0.96–1.07 of its earlier value in all 7 tests. It also gives clearly fewer ranking errors on TRN, R3D-18 and R3D-50 (non-overlapping 95% intervals), and its ranking errors don't increase when a segment of a V-JEPA2 clip is sampled more densely. On MC3-18, ranking errors are rare for every method.
2. **Fixed windows are not a remedy.** They split repeated content across windows and dilute almost as much as frame-level Shapley. That holds with the same number of groups as DINOv2, and more so with windows of a fixed width. This answers "why not just use windows?"
3. **The grouping has to adapt to the sampling density.** A fixed number of groups works with copies but not fully with denser sampling. A threshold adapts, but only with a linkage whose merge height doesn't grow with cluster size (average, not Ward).
4. **The remedy is coarser, but that isn't why it works, and it's cheaper.** It uses about 2.5 frames per group, close to 8 windows with 2 frames each, which still dilute. And it needs 2–35 times fewer model evaluations than frame-level Shapley, with exact values.

## 6. Limitations

- **The per-copy readout depends on group membership.** For a grouping that adapts to the clip, the target's group in the reference clip can contain different frames from its group in the target clip. On V-JEPA2's E1, `dino_k4` and `dino_thr` drop to 0.60–0.66 when the reference group included neighbouring frames, and to 0.82–0.92 when the target was alone in it. So part of their drop comes from the membership change, not from dilution. `dinoavg_thr` stays at 0.87–0.99 in both cases.
- **The 8-group cap.** On MC3-18's and V-JEPA2's E1 clips (16–24 frames), the threshold groupings hit the cap in 41–73% of clips, and then behave like `dino_k8`. The cap only limits clips that would have more groups, not the merging of similar frames.
- **Static clips.** With average linkage, a nearly static clip can merge completely. The floor of 2 groups was reached in 7–18% of clips.
- **Top-1 loss** rests on few cases on R3D-50, MC3-18 and V-JEPA2.
- **Ties favour coarse groups.** With broadcast scores, frames in one group share a score, and a tie doesn't count as an inversion. A grouping that merges the target with other frames avoids some inversions by construction. For `dinoavg_thr`, 75–100% of the target's group are its copies, so the effect is limited. A measure that doesn't depend on ties, such as deletion and insertion curves, would settle this.
- **MC3-18's window results are confounded by clip length.** Insertion makes the clip longer, so the control clips also get more windows, and much of `win_w4`'s drop (−0.18) also appears in the control (−0.14).
- **Repeated frames at d = 3** in the V-JEPA2 frame-rate test, from short SSv2 videos (see `results/group_attribution_summary.md`).
- **Group values don't follow a model that counts copies** (MC3-18's I_t ratio is 2.88), as noted in the step-2 summary.
- **One τ per model**, calibrated for a median of n/2 groups. Other targets would change the resolution of the explanation. We haven't tested how sensitive the results are to τ.

## 7. Files and how to reproduce

| file | what it contains |
|---|---|
| `results/e1/dino_tau.json` | τ per model and linkage, with the calibration details |
| `results/e1/<model>_group_attributions.jsonl` | group values per clip, now also for the practical groupings (`info` records the number of groups, the cap and floor, and where the target's copies ended up) |
| `results/e1/<model>_practical_{rows,slopes,summary,beta}.csv`, `_practical_metrics.log` | E1 readouts |
| `results/fr/<model>_fr_practical.jsonl`, `_fr_practical_{rows,summary}.csv`, `_metrics.log` | frame-rate test data and readouts |
| `results/practical_extras_{sizes,ranks,cost}.csv` | section 4: group sizes, ranking readouts with 95% CIs, and evaluations per clip |
| `results/kaggle_logs/practical_job*` | run logs |

Commands (TRN and the MC3-18 frame-rate test ran on the laptop; the rest on Kaggle, 2 × T4, with `--shard s --num-shards 2`):
```
python motivation/calibrate_tau.py --models r3d r3d_18 mc3_18 vjepa2 [--linkage average]
python motivation/calibrate_tau.py --models trn_official --from-videos 150 [--linkage average]

G="dino_k4 dino_k8 dino_thr dinoavg_thr win_k4 win_k8"
python motivation/e1_group.py --model trn_official --fp16 --dup-types exact --groupings dino_k2 dino_k4 dino_thr dinoavg_thr win_k2 win_k4 win_w2
python motivation/e1_group.py --model mc3_18 --fp16 --limit 60 --dup-types exact --max-groups 8 --groupings $G win_w4
python motivation/e1_group.py --model r3d_18 --fp16 --limit 60 --dup-types exact --max-groups 8 --groupings $G
python motivation/e1_group.py --model r3d    --fp16 --limit 60 --dup-types exact --max-groups 8 --groupings $G
python motivation/e1_group.py --model vjepa2 --fp16 --dup-types exact --max-groups 8 --groupings dino_k4 dino_thr dinoavg_thr win_k4 win_w4
python motivation/fr_practical.py --model mc3_18 --fp16
python motivation/fr_practical.py --model vjepa2 --fp16 --max-groups 8

python motivation/e1_metrics.py attr --model <model> --practical
python motivation/fr_metrics.py --model <model> --name fr_practical
python motivation/practical_extras.py      # section 4, from the files above; no model runs
python motivation/test_e23.py
```
