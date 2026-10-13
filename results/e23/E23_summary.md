# E2/E3 summary: attribution dilution in natural videos

## In short

E1 showed dilution with copies we made ourselves. E2/E3 ask whether it also happens in real videos, where many frames already look alike.

**What we mean by dilution.** When the evidence for a prediction is spread over several frames that can stand in for each other, a frame-level method divides the credit among them. Each frame then looks unimportant, and that part of the video can rank below a part shown in fewer frames, even when the model relies on it more.

**Main results** (four models, 554 videos):
1. **Natural clips are highly redundant for the models.** If you remove frames one at a time, you see almost none of what the clip contributes: the single-frame importances add up to about 0% of the clip's importance. Removing the 4 top-ranked frames still leaves 75–95% of it.
2. **Leave-one-out and occlusion lose redundant evidence.** The more the frames of a group can stand in for each other, the less credit these methods give the group. This is the same failure as in E1.
3. **Dilution misleads the reading of frame scores.** When the more important part of a clip is spread over many frames, a user reading Shapley's frame scores ranks it below a less important, shorter part much more often:
   - V-JEPA2: 42% of the time, against 25% when both parts are the same size;
   - R3D-50: 28% against 15%;
   - R3D-18: 29% against 19%.

   These differences are significant and hold for both random seeds. MC3-18 shows no effect, because there the whole clip is redundant.

Some first versions of the readouts didn't work. They're described in section 7, with the reasons.

## 1. The question

Real videos often contain runs of frames that look almost the same: a static shot, a slow movement, a still camera. If dilution is real, then a part of the video shown in many such frames should get its credit divided, and look less important than it is.

## 2. How the test works

**Models and data.** The four models from E1 that can be tested, with the same frame sampling and the same way of removing frames:

| model | data | removing a frame | videos used (screened) |
|---|---|---|---|
| MC3-18 | UCF101 test | delete it | 169 (200) |
| R3D-18 | UCF101 test | fill it with a neighbour (freeze) | 159 (200) |
| R3D-50 | UCF101 test | freeze | 158 (200) |
| V-JEPA2 | SSv2, full test set (27,157 videos) | delete it | 68 (100) |

The videos are sampled evenly across classes. The others were misclassified and skipped. Everything explains the predicted class.

**For each video** (`motivation/e23_run.py`):
1. Take the model's own 16 frames, and predict.
2. **Group similar frames:** embed each frame with DINOv2, and cluster. Only neighbouring frames can merge, so each group is a run of consecutive, similar frames. This gives a tree, from 16 single frames up to the whole clip.
3. **Measure, with the model only:** each frame's own importance (remove just that frame), and each group's importance (remove the whole group). Importance is the drop in the predicted class's probability.
4. **Run the attribution methods** on the normal clip: Shapley (64 orderings, 2 seeds), leave-one-out, occlusion, Integrated Gradients and Grad-CAM. Each frame gets a score.
5. Record deletion curves and a swap test (section 6).

**For the dilution test** (`motivation/e23_group.py`), we also measure each part of the clip **as a unit**. We cut the tree into k segments (k = 4 and k = 8), and compute exact Shapley values with **each segment as one player**. No credit is divided inside a segment, so this value isn't diluted: it says how much the segment matters as a whole.

## 3. Natural clips are highly redundant

These numbers come from the model alone, with no attribution method. They use videos where the clip matters (probability of the predicted class minus the prior > 0.05; all videos qualify).

| | MC3-18 | R3D-18 | R3D-50 | V-JEPA2 |
|---|---|---|---|---|
| single-frame importances added up ÷ clip importance | −0.04 | −0.07 | 0.00 | 0.02 |
| most important single frame ÷ clip importance | 0.02 | 0.06 | 0.02 | 0.06 |
| clip importance left after removing the top 1 / 2 / 4 frames (best method) | 0.98 / 0.96 / 0.93 | 0.94 / 0.91 / 0.89 | 0.98 / 0.96 / 0.95 | 0.94 / 0.86 / 0.75 |

Medians over videos. "Clip importance" is the probability of the predicted class minus the prior: what the clip as a whole contributes.

What this shows:
- **Removing one frame at a time reveals almost nothing.** Added up, the single-frame importances explain about 0% of what the clip contributes. The model can do without any single frame, because other frames show the same thing.
- **Even the best-ranked frames can be removed.** After removing the 4 frames a method ranks highest, 75–95% of the clip's contribution is still there.
- **V-JEPA2 depends a little more on particular frames**, which fits SSv2, where classes depend on motion.

**Redundancy of a group.** For a group G that matters (its importance I(G) > 0.02), we measure how much of its importance single-frame removal can't see:

R(G) = 1 − (sum of the single-frame importances in G) ÷ I(G)

R ≈ 1 means no single frame of the group is needed (they stand in for each other). R ≈ 0 means each frame adds its own part. Median R over groups that matter:

| group size | MC3-18 | R3D-18 | R3D-50 | V-JEPA2 |
|---|---|---|---|---|
| 2 frames | 0.15 | 0.35 | 0.37 | 0.68 |
| 3–4 frames | 0.30 | 0.58 | 0.70 | 0.61 |
| 5–8 frames | 0.56 | 0.83 | 0.81 | 0.93 |

(DINOv2 groups; fixed windows of the same size give similar values.) The bigger the group, the more of its importance is invisible frame by frame.

## 4. Leave-one-out and occlusion lose redundant evidence

For each group that matters, we compare a method's credit for the group's frames, added up, with the group's importance (r_sum). Medians, by redundancy:

| redundancy R | < 0.25 | 0.25–0.5 | 0.5–0.75 | ≥ 0.75 |
|---|---|---|---|---|
| R3D-18, occlusion | 1.01 | 0.68 | 0.27 | 0.08 |
| R3D-50, occlusion | 1.95 | 1.31 | 0.59 | 0.75 |
| V-JEPA2, occlusion | 1.10 | 0.57 | 0.76 | 0.50 |
| MC3-18, occlusion | 2.74 | 2.43 | 2.28 | 2.06 |
| R3D-18, Shapley | 2.65 | 2.21 | 2.13 | 2.84 |

- **Occlusion gives redundant groups less and less credit.** Blanking one frame changes little when its neighbours show the same thing. On R3D-18, a highly redundant group keeps only 8% of its importance. This is E1's leave-one-out and occlusion failure, now with natural near-duplicates.
- **Leave-one-out** falls in the same way, but for leave-one-out this partly follows from how R is defined. So occlusion is the independent evidence.
- **Shapley, Integrated Gradients and Grad-CAM don't lose the group's total.** Their failure is a different one: they divide the credit among the frames. Section 5 tests that.

## 5. Dilution: the frame scores mislead about which part matters more

**The test.** In each clip cut into k segments, take pairs of segments (A, B) where A matters clearly more than B as a unit. Specifically, A's group value is more than 0.02 above B's, and more than 0.02 in absolute terms. Then read the frame scores as a user would: **do B's frames have a higher average score than A's?** If so, a user would conclude that B matters more. We call that an *inversion*.

We compare:
- **pairs where A is at least twice as large as B.** A's credit is divided over many more frames, so dilution predicts more inversions here;
- **pairs where A and B have about the same size** (within a factor of 2). These are the comparison.

**Chance level.** We repeat the same comparisons with each video's frame scores rotated by every possible offset. That keeps how smooth the scores are over time, but breaks their link to the segments. For the average score, chance is about 0.5 in both kinds of pairs. *Gain* = chance − observed inversion rate: how much better than chance the scores rank the parts. The 95% confidence intervals come from resampling videos.

**Shapley, average credit per frame:**

| model | k | inversions, A ≥ 2× larger | inversions, same size | gain difference [95% CI] | pairs (larger / same) |
|---|---|---|---|---|---|
| V-JEPA2 | 4 | **0.42** | 0.25 | **−0.20 [−0.33, −0.06]** | 125 / 172 |
| V-JEPA2 | 8 | **0.32** | 0.19 | **−0.14 [−0.23, −0.06]** | 619 / 632 |
| R3D-50 | 4 | 0.27 | 0.23 | −0.05 [−0.15, +0.06] | 230 / 213 |
| R3D-50 | 8 | **0.28** | 0.15 | **−0.15 [−0.20, −0.10]** | 1,046 / 913 |
| R3D-18 | 4 | **0.29** | 0.19 | **−0.12 [−0.19, −0.04]** | 313 / 324 |
| R3D-18 | 8 | **0.16** | 0.11 | **−0.07 [−0.10, −0.03]** | 1,213 / 1,195 |
| MC3-18 | 4 | 0.18 | 0.11 | −0.05 [−0.13, +0.02] | 158 / 285 |
| MC3-18 | 8 | 0.12 | 0.12 | +0.01 [−0.03, +0.05] | 653 / 977 |

A negative gain difference means the frame scores lose more of their ranking ability when the more important part is the larger one. That's dilution.

**With the second Shapley seed:** V-JEPA2 −0.19 and −0.20, R3D-50 −0.10 and −0.14, R3D-18 −0.10 and −0.09, MC3-18 −0.02 and −0.02. Same picture.

What this shows:
- **On V-JEPA2, R3D-50 and R3D-18, Shapley's frame scores mislead about which part of the clip matters more**, when that part is spread over many frames. On V-JEPA2 with 4 segments, the user picks the wrong part 42% of the time, against 25% when the parts are the same size.
- **MC3-18 shows no effect.** Its segments all have about the same group value: any part of the clip can stand in for any other (MC3-18 recognises 78% of UCF101 videos from a single frame). So there's no clearly more important part for dilution to hide.

**Other methods** (average credit per frame, gain difference):

| method | V-JEPA2, k = 4 / 8 | R3D-50, k = 4 / 8 | R3D-18, k = 4 / 8 | MC3-18, k = 4 / 8 |
|---|---|---|---|---|
| Occlusion | **−0.14 / −0.16** | +0.04 / +0.01 | −0.04 / −0.01 | −0.06 / **−0.05** |
| Leave-one-out | +0.00 / −0.03 | +0.05 / −0.05 | **−0.11 / −0.08** | −0.04 / −0.02 |
| Integrated Gradients | −0.06 / −0.04 | −0.05 / −0.01 | **−0.13** / −0.03 | −0.03 / +0.00 |
| Grad-CAM | +0.06 / +0.01 | −0.02 / −0.01 | **−0.21** / −0.05 | −0.09 / **−0.07** |

Bold: the 95% confidence interval excludes 0. Shapley shows the most consistent effect. Occlusion shows it clearly on V-JEPA2.

**The best frame instead of the average.** If the user looks at each part's best frame rather than its average, a longer part gets a boost by chance alone: the best of more frames tends to be higher. Measured against its own chance level, the best-frame readout points the same way: Shapley's gain difference is −0.08 to −0.26 on V-JEPA2, R3D-50 and R3D-18 (all but R3D-50 at k = 4 exclude 0). But because the chance levels differ between the two kinds of pairs (about 0.3 against 0.5), the average-per-frame readout is the cleaner one.

## 6. Do DINOv2 groups match how the model sees the frames?

**The swap test.** For each pair of neighbouring frames, we replace one with the other and measure how much the output changes.

| | MC3-18 | R3D-18 | R3D-50 | V-JEPA2 |
|---|---|---|---|---|
| rank correlation: DINOv2 similarity vs output change (JS) | −0.47 | −0.49 | −0.40 | −0.11 |
| mean change in P(class), swap within a k = 8 group / across groups | 0.009 / 0.011 | 0.021 / 0.021 | 0.020 / 0.022 | 0.039 / 0.055 |

- **On the UCF101 models, more similar frames are more interchangeable for the model**, but within-group and across-group swaps change the output about equally. With 16 frames spread over a whole clip, almost all neighbouring frames are similar (mean DINOv2 similarity 0.92–0.95).
- **On V-JEPA2, DINOv2 similarity predicts interchangeability only weakly.** SSv2 depends on small motions, which DINOv2 doesn't capture well. But swaps within a group do change the output less (0.039 against 0.055).

## 7. What didn't work, and why

- **The first E2 readout** compared how the best frame's share of its group's importance changes with group size, for DINOv2 groups vs fixed windows. It came out opposite to the prediction. The readout was flawed: a window's importance grows with its size because it holds more distinct content, while a run of similar frames gains little. So the slope measured how importance grows with size, not credit division. It's kept in the output files, marked as confounded.
- **The first E3 inversion test** ranked groups by "remove the whole group" importance. That value shrinks when the rest of the clip can compensate, and grows with group size. It also compared best frames without a chance baseline. Both bias it against finding dilution. It's replaced by the test in section 5, which uses group Shapley values and a chance baseline.
- **DINOv2 groups vs fixed windows.** At 16 frames per clip, DINOv2 groups aren't more redundant than fixed windows of the same size (section 3), because nearly all neighbouring frames are similar. So E2/E3 can't show that similarity-based grouping is better than simple windows. The planned frame-rate test (denser sampling of the same time window) is where this should be tested.

## 8. Conclusions

1. **Natural video clips are highly redundant for all four models.** Removing frames one at a time reveals almost none of what the clip contributes. Dilution has real material to act on.
2. **Removal-based methods lose redundant evidence.** Occlusion's credit for a group falls as the group's redundancy rises, as E1 showed with exact copies.
3. **Dilution misleads the reading of frame scores in real videos.** When the more important part of a clip is spread over many frames, Shapley's frame scores rank it below a less important, shorter part significantly more often, on V-JEPA2, R3D-50 and R3D-18. On V-JEPA2 the error rate rises from 25% to 42%.
4. **When the whole clip is redundant (MC3-18 on UCF101), there's no part for dilution to hide.** Frame scores are flat, and so are the segment values. Here dilution shows up as "nothing stands out" rather than as wrong rankings.
5. **Group-level values avoid the problem by construction.** Shapley over segments doesn't divide credit inside a segment. A test on the toy model confirms the property: a segment's group value stays exactly the same when its run of copies grows from 2 to 5, while the frame-level credit per copy falls from 0.0094 to 0.0015. This is the basis for the method part of the paper.

## 9. Limitations

- **Which segments.** The segments come from the DINOv2 tree, at two levels (4 and 8 segments). Other groupings could give different pairs.
- **Group Shapley as the yardstick.** It measures each part's value as a unit, given the other parts. It's the natural yardstick for "which part matters more", but it's not a value-free ground truth.
- **Freeze removal on the R3D models.** Filling a removed frame with a near-identical neighbour can make single frames look less necessary. MC3-18 and V-JEPA2 delete frames instead, and V-JEPA2 shows the clearest effect, so the main result doesn't depend on freeze.
- **Sample sizes.** 68 videos for V-JEPA2, 158–169 for the others. The confidence intervals account for this, by resampling videos.
- **Shapley noise.** Two seeds of the same video agree at 0.83 (R3D-18), 0.64 (V-JEPA2), 0.49 (R3D-50) and 0.46 (MC3-18) per frame. Each video uses its own random orderings, so the noise isn't shared across videos.

## 10. Files and how to reproduce

Files in `results/e23/`:

| file | what it contains |
|---|---|
| `<model>_e23.jsonl`, `<model>_e23_skipped.jsonl` | per video: clusters, importances, attributions, deletion curves, swap test; and the skipped videos |
| `<model>_e23_group.jsonl` | group Shapley values per segment, at k = 4 and 8 |
| `<model>_e2_visibility.csv`, `_e2_redundancy.csv`, `_e2_method_vs_R.csv` | sections 3 and 4 |
| `<model>_dil_pairs.csv`, `_dil_contrast.csv`, `_dil_top.csv` | section 5 |
| `<model>_swap.csv` | section 6 |
| `<model>_e2_rows.csv`, `_e2_bins.csv`, `_e2_slopes.csv`, `_e3_*.csv` | the first readouts (section 7) |
| `<model>_metrics.log` | all tables, as printed |
| `ucf.log`, `vjepa2.log`, `<model>_group.log` | run logs |

Commands:
```
# per-video data (ran on Kaggle, 2 x T4)
python motivation/e23_run.py --model mc3_18 --limit 200 --fp16 --out results/e23
python motivation/e23_run.py --model r3d_18 --limit 200 --fp16 --out results/e23
python motivation/e23_run.py --model r3d    --limit 200 --fp16 --out results/e23
python motivation/e23_run.py --model vjepa2 --dataset ssv2 --limit 100 --fp16 --n-perm 32 --batch-size 64 --out results/e23

# segment values (laptop; ~1-3 s per video for UCF101 models, ~18 s for V-JEPA2)
python motivation/e23_group.py --model <model> --fp16 --out results/e23

# all readouts
python motivation/e23_metrics.py --model <model> --out results/e23

# tests
python motivation/test_e23.py
```
