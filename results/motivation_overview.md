# Attribution dilution in video: overview of all experiments

This report brings together everything we have tested about attribution dilution so far. It gives the question, the method and the main numbers for each experiment, and links to the detailed reports.

| detailed report | covers |
|---|---|
| `results/E1_report.md` and `results/e1/*_E1_summary.md` | E1, controlled duplication, on all models and per model |
| `results/design_choice_report.md` | which models can be used, and why |
| `results/e23/E23_summary.md` | E2 (redundancy in real clips) and E3 (the dilution test in real clips) |
| `results/group_attribution_summary.md` | group values on the E1 clips, and the frame-rate test |
| `results/practical_grouping_summary.md` | groupings chosen without knowing which frames are redundant: DINOv2 against fixed windows |

## In short

1. **Dilution is real, and duplication causes it** (E1). When an important frame is shown several times, the model relies on it about as much as before (0.87–1.32 times, on four of five models). But Shapley gives each copy only 0.16–0.62 of the frame's earlier credit, and leave-one-out about 0, on all five models. Integrated Gradients and occlusion do the same on most models.
2. **Real clips are highly redundant** (E2). Removing frames one at a time reveals almost nothing: the single-frame importances add up to about 0% of what the clip contributes. Removing the 4 top-ranked frames still leaves 75–95% of it.
3. **Dilution misleads users in real videos** (E3). When the more important part of a clip is spread over many frames, Shapley's frame scores rank it below a less important, shorter part significantly more often. On V-JEPA2: 42% of the time, against 25% for parts of the same size.
4. **It happens with denser real frames too** (frame-rate test). On V-JEPA2, sampling the most important quarter of a clip twice as densely leaves the model's reliance on it unchanged (0.96). But each of its frames gets 0.67 of its earlier Shapley credit.
5. **Giving credit to groups removes it** (group follow-up). Shapley with all copies of a frame as one player keeps the group's credit at 0.94–1.12 of its earlier value on all five models. DINOv2 clustering finds those groups in 86–100% of clips.
6. **It works without knowing which frames are redundant** (practical groupings). DINOv2 similarity with a fixed average-linkage threshold, calibrated once per model on other videos, keeps repeated content's credit at 0.96–1.07 of its earlier value in all 7 tests (E1 on five models, the frame-rate test on two). Frame-level Shapley gives 0.16–0.62.
7. **Fixed windows are not a remedy.** They split repeated content across windows. With the same number of groups as DINOv2, they give it 0.35–0.66 of its earlier credit (0.83–0.90 on MC3-18). Windows of a fixed width give 0.26–0.65 on the models whose clips get longer. Both are about as diluted as frame-level Shapley.

## 1. What we mean by attribution dilution

A video often shows the same evidence in several frames: a still shot, a slow movement, a static background. Those frames can stand in for each other: removing one of them barely changes the model's output, because the others still show the same thing.

A frame-level attribution method gives each frame its own score. When frames can stand in for each other, the method divides the evidence's credit among them. For Shapley values this is forced by the symmetry rule: interchangeable players get equal shares. The result:
- **each frame looks unimportant,** even though the evidence matters;
- **that evidence can rank below evidence shown in fewer frames,** even when the model relies on it more.

Dilution doesn't need the credits to add up wrongly. Shapley's credits always add up to the clip's total. Dividing them is exactly what makes each frame look small.

**The yardstick is the model itself.** A content's *importance* I is the drop in the predicted class's probability when every frame showing it is removed. All experiments compare a method's credit with I, or with a group's Shapley value, never with an ideal answer.

## 2. Models and data

| model | data | design in E1 | how methods remove a frame | used in |
|---|---|---|---|---|
| R3D-50 | UCF101 | reallocation | fill with a neighbouring frame (freeze) | E1, E2/E3, group, practical |
| R3D-18 | UCF101 | reallocation | freeze | E1, E2/E3, group, practical |
| MC3-18 | UCF101 | insertion | delete | E1, E2/E3, group, frame-rate, practical |
| V-JEPA2 | SSv2 | insertion | delete | E1, E2/E3, group, frame-rate, practical |
| TRN | SSv2 | reallocation (8 frames) | freeze | E1, group, practical |

- **Reallocation** is for models that need a fixed clip length. Copies of the target take slots from other frames.
- **Insertion** is for models that accept other lengths. Copies are added, and the clip gets longer.

**Not used:**
- **VideoMAE.** It reads frames in pairs, and two identical frames in a pair change its output, whatever they show. So duplicating a frame changes the model itself (`results/e1/VIDEOMAE_E1_summary.md`).
- **Play Fair's own TRN.** Its checkpoints are no longer available.

**Methods:** Shapley, Play Fair, leave-one-out, occlusion, Integrated Gradients and Grad-CAM.

## 3. E1: controlled duplication

**Question.** When a frame the model relies on is shown m times, does a method still give it the right credit?

**How.**
- Pick a *target* frame by the model's own importance I.
- Build a **target clip**, where the target is shown m times.
- Build a **control clip**, which makes the same change but gives the extra copies to other frames.
- Compare the target's credit in each clip with its credit in a reference clip.
- A change that appears in the target clip but not in the control is caused by duplicating the target.

**Readouts:**
- **I_t ratio:** the model's reliance on the target, relative to the reference.
- **Per-copy ratio:** the credit of one copy.
- **Total ratio:** the credit of all copies added up.
- **β:** how fast the per-copy credit falls with m. β = 0 means no change, and β = −1 means the credit is split evenly among the copies.
- **Ranking:** top-1 loss and inversions.

**Main numbers** (exact copies, highest m; LOO = leave-one-out):

| | R3D-50 (×4) | R3D-18 (×4) | MC3-18 (×9) | V-JEPA2 (×9) | TRN (×2) |
|---|---|---|---|---|---|
| model's reliance (I_t ratio) | 1.28 | 1.30 | 3.22 | 0.87 | 1.32 |
| Shapley, credit of one copy | 0.62 | 0.52 | 0.48 | 0.16 | 0.50 |
| Shapley, all copies added up | 2.47 | 2.07 | 4.33 | 1.45 | 0.99 |
| Shapley, β, target (control) | −0.37 (−0.01) | −0.47 (+0.03) | −0.31 (−0.12) | −0.71 (−0.02) | −0.94 (−0.03) |
| LOO, credit of one copy | −0.05 | −0.02 | 0.13 | 0.30 | 0.02 |
| LOO, β, target (control) | −1.38 (−0.11) | −1.55 (−0.09) | −0.77 (−0.14) | −0.48 (−0.05) | −2.60 (+0.25) |
| Shapley, top-1 loss, target vs noise alone | 0.71 vs 0.47 | 0.75 vs 0.38 | 0.24 vs 0.62 | too few cases | 0.61 vs 0 |

**What it shows:**
- **Shapley and leave-one-out dilute on all five models, and duplication causes it.** The target's β is clearly negative, and far below the control's. The controls are near 0, except on MC3-18 (−0.12 to −0.14, following the model's own change of −0.14 in the longer control clip) and LOO on TRN (+0.25).
- **Integrated Gradients and occlusion dilute on most models.** The exceptions are occlusion on R3D-50, which shows no effect specific to duplication, and Integrated Gradients on V-JEPA2, whose credit also falls in the control.
- **Grad-CAM barely dilutes.** Its credit per copy changes little, and on three models the control changes as much. Its problem is too much credit in total (below).
- β below −1 means the credit fell to almost 0. LOO does this on three models.
- **Leave-one-out gives copies almost no credit** (0.13 or less per copy on four models). Removing one copy leaves the others, so the output barely changes.
- **Dilution changes rankings.** With Shapley, the duplicated target loses its top rank in 75% of cases on R3D-18 and 61% on TRN.
- **A second, separate failure: too much credit in total.** Grad-CAM on every model, and Shapley on four models, give the copies together more credit than the model's reliance supports.
- **Play Fair gives the same values as Shapley-drop,** within random noise, on two models. So the Shapley-drop results apply to Play Fair.
- **Exact copies aren't needed.** Copies with slight noise give the same results.

## 4. E2: how redundant are real clips?

**Question.** Real clips already contain near-identical frames. How much do the models' outputs depend on any single frame?

**How.**
- Take the model's own 16 frames.
- Cluster them with DINOv2 into runs of similar neighbouring frames.
- Measure, with the model only:
  - each frame's importance (remove just that frame);
  - each group's importance (remove the whole group).
- **Redundancy** R(G) = 1 − (sum of the single-frame importances in G) ÷ I(G). R near 1 means removing frames one at a time can't see the group's importance.

Four models, 554 videos.

**Main numbers** (medians over videos):

| | MC3-18 | R3D-18 | R3D-50 | V-JEPA2 |
|---|---|---|---|---|
| single-frame importances added up ÷ clip importance | −0.04 | −0.07 | 0.00 | 0.02 |
| clip importance left after removing the top 4 frames | 0.93 | 0.89 | 0.95 | 0.75 |
| redundancy R of 5–8 frame groups | 0.56 | 0.83 | 0.81 | 0.93 |

**What it shows:**
- **Natural clips are highly redundant for all four models.** No single frame is needed, because other frames show the same thing.
- **Occlusion and leave-one-out lose redundant evidence.** The more redundant a group, the less credit occlusion gives it. On R3D-18, highly redundant groups (R ≥ 0.75) keep only 8% of their importance.

## 5. E3: does dilution mislead users in real clips?

**Question.** A user reads frame scores to decide which part of a clip matters more. Does that go wrong when the more important part is spread over more frames?

**How.**
- Cut each clip into k segments (k = 4 and 8) using the DINOv2 tree.
- Compute **group Shapley** with each segment as one player. No credit is divided inside a segment, so this value is the yardstick for "which part matters more".
- For pairs of segments (A, B) where A clearly matters more, count an **inversion** when B's frames have a higher average frame score than A's.
- Compare pairs where A is at least twice as large as B with pairs of about the same size.
- A chance baseline comes from rotating each video's scores. 95% confidence intervals come from resampling videos.

**Main numbers** (Shapley, average score per frame):

| model | k | inversions, A ≥ 2× larger | inversions, same size | gain difference [95% CI] |
|---|---|---|---|---|
| V-JEPA2 | 4 | 0.42 | 0.25 | −0.20 [−0.33, −0.06] |
| V-JEPA2 | 8 | 0.32 | 0.19 | −0.14 [−0.23, −0.06] |
| R3D-50 | 8 | 0.28 | 0.15 | −0.15 [−0.20, −0.10] |
| R3D-18 | 4 | 0.29 | 0.19 | −0.12 [−0.19, −0.04] |
| MC3-18 | 8 | 0.12 | 0.12 | +0.01 [−0.03, +0.05] |

A negative gain difference means the frame scores rank the parts worse when the more important part is the larger one.

**What it shows:**
- **On V-JEPA2, R3D-50 and R3D-18, Shapley's frame scores mislead** when the more important part is spread over many frames. The result holds for both Shapley seeds.
- **MC3-18 shows no effect.** Its segments all have about the same value, so there's no clearly more important part to hide.

## 6. The frame-rate test: dilution with real frames

**Question.** In E1 we made the copies. Does dilution also happen when the extra frames are real frames from the same stretch of the video?

**How.**
- Split the model's 16 frames into 4 time segments.
- Take the most important segment, by group Shapley, as the **target**, and the least important as the **control**.
- Sample the target segment 2 or 3 times as densely, using frames from the same stretch of the video. Do the same with the control segment in a separate clip.
- Only models that accept other clip lengths can be used: V-JEPA2 (29 videos) and MC3-18 (48 videos).

**Main numbers** (V-JEPA2, target segment, relative to the base clip):

| readout | d = 2 | d = 3 |
|---|---|---|
| model's reliance on the segment | 0.96 | 0.94 |
| group Shapley value of the segment | 1.00 | 0.98 |
| Shapley credit per frame | **0.67** | **0.49** |
| inversions with frame Shapley (base clip: 0.26) | 0.23 | 0.41 |

**What it shows:**
- **Dilution happens with real, distinct frames.** The model relies on the denser segment as before, but each of its frames gets less Shapley credit (β −0.50 [−0.84, −0.29]; control +0.06).
- **It isn't caused by repeated frames.** On the 25 clips at d = 2 where every frame is distinct, the per-frame credit is also 0.67. At d = 3, many SSv2 videos are too short and some frames repeat.
- **On MC3-18 the effect is weaker,** and the ranking doesn't change. As in E3, its segments matter about equally.

## 7. Group values remove dilution

**Question.** If credit goes to groups of frames instead of single frames, does dilution go away?

**How.** On the E1 clips, compute Shapley with each group as one player. The groupings:
- **oracle:** all copies of a content form one group;
- **DINOv2:** clustering with the same number of groups;
- **DINOv2 with half as many groups;**
- **pairs (V-JEPA2 only):** the target's copies form one group, and the other frames are joined into neighbouring pairs. This keeps the number of groups at 9, so exact Shapley is possible.

**Main numbers** (target clips, exact copies, highest m):

| | R3D-50 (×4) | R3D-18 (×4) | TRN (×2) | MC3-18 (×9) | V-JEPA2 (×9) |
|---|---|---|---|---|---|
| model's reliance (I_t ratio) | 1.01 | 1.27 | 1.32 | 2.88 | 0.87 |
| frame Shapley, one copy | 0.61 | 0.52 | 0.49 | 0.54 | 0.14 |
| **group Shapley** | **1.09** | **1.06** | **1.12** | **1.02** | **0.94** |
| β, frame per copy | −0.35 | −0.46 | −0.94 | −0.24 | −0.64 |
| β, group | +0.06 | +0.05 | +0.17 | +0.09 | +0.04 |
| top-1 loss, frame vs group | 0.50 vs 0.19 | 0.64 vs 0.00 | 0.61 vs 0.04 | too few cases | too few cases |

The I_t ratios differ a little from section 3, because this run uses a subset of E1's videos for R3D-50, R3D-18, MC3-18 and V-JEPA2.

**What it shows:**
- **Group values don't dilute** on any model, while frame Shapley's per-copy credit falls on every model.
- **Group values keep the ranking** on R3D-18 and TRN. On R3D-50, frame scores change the top rank as often in the control clips, so that result is weak. On MC3-18 and V-JEPA2 there are too few top-ranked targets to read.
- **DINOv2 finds the copies.** At the matched number of groups, it puts them in one group in 86–100% of clips, and then gives the same values as the oracle. Coarser groups stay undiluted, but they mix the target with other frames.
- **On V-JEPA2, the exact group values follow the model.** Their rank correlation with the I_t ratio is 0.72, against 0.53 for frame credit per copy.
- **Limitation:** group values stay flat even when the model gives repeated frames more weight. On MC3-18, the model relies on the target 2.9 times as much, but the group value stays at 1.0.

## 8. Practical groupings: no knowledge of the copies

**Question.** Section 7 used the true copies, or DINOv2 cut to the true number of groups. Neither is known for a new video. Does a grouping chosen in advance still remove dilution, and do similarity-based groups do better than simple fixed windows?

**How** (`results/practical_grouping_summary.md`).
- **Groupings, all fixed per model before the test:**
  - DINOv2 cut into a fixed number of groups K (`dino_k4`, `dino_k8`);
  - DINOv2 cut at a fixed similarity threshold τ, with Ward linkage (`dino_thr`) or average linkage (`dinoavg_thr`);
  - fixed windows: K equal windows (`win_k4`, `win_k8`), or windows of a fixed width (`win_w4`).
- **τ** is calibrated once per model on other videos, so that the median clip has half as many groups as frames. The threshold groupings keep between 2 and 8 groups.
- **Readout:** group Shapley, with every frame showing its group's value, not divided ("broadcast scores"). This is scored exactly like a frame-level method, against the model's own reliance, on the E1 clips (all five models) and the frame-rate test (MC3-18, V-JEPA2).

**Main numbers** (credit per copy, or per frame, at the highest level, relative to the reference; β in brackets):

| | TRN, E1 | R3D-18, E1 | R3D-50, E1 | MC3-18, E1 | V-JEPA2, E1 | MC3-18, frame-rate | V-JEPA2, frame-rate |
|---|---|---|---|---|---|---|---|
| frame Shapley | 0.50 (−0.94) | 0.54 (−0.45) | 0.62 (−0.36) | 0.49 (−0.30) | 0.16 (−0.71) | 0.62 (−0.44) | 0.49 (−0.50) |
| **`dinoavg_thr`** | **1.07 (+0.12)** | **1.04 (+0.05)** | **1.05 (+0.04)** | **1.05 (+0.03)** | **0.97 (−0.01)** | **0.98 (−0.02)** | **1.04 (+0.04)** |
| `dino_thr` (Ward) | 1.13 | 1.23 | 1.19 | 0.96 | 0.73 | 0.77 | 0.63 |
| DINOv2, fixed K | 1.12 | 1.07 | 1.09 | 0.97 | 0.72 | 0.94–0.98 | 0.79–0.89 |
| windows, same K as DINOv2 | 0.58 | 0.58 | 0.66 | 0.83 | 0.35 | 0.90 | 0.59 |
| windows, fixed width | 0.58 | = same K | = same K | 0.65 | 0.26 | 0.61 | 0.37 |

TRN, R3D-18 and R3D-50 keep a fixed clip length, so their fixed-width windows are the same as `win_k4`.

**Ranking** (with 95% confidence intervals from resampling videos). On TRN, R3D-18 and R3D-50, `dinoavg_thr`'s inversions in the E1 target clips fall to 1–4%, against 13–36% for frame Shapley and 25–33% for windows, and the intervals don't overlap. On V-JEPA2's E1 they fall from 19% to 8%, with overlapping intervals. On MC3-18, inversions are rare for every method (2–8%). In the V-JEPA2 frame-rate test, `dinoavg_thr`'s inversions don't rise when the segment gets denser (0.23, against 0.26 in its control), while every other method's rise to 0.38–0.42.

**Resolution and cost.** `dinoavg_thr` uses about 2.5 frames per group, close to 8 windows with 2 frames each, which still dilute; so the coarser resolution isn't what removes dilution. It needs 2–35 times fewer model evaluations than frame-level Shapley, and its values are exact.

**What it shows:**
- **DINOv2 groups at a fixed average-linkage threshold remove dilution in all 7 tests,** without knowing which frames are redundant. Every DINOv2 grouping puts all copies of the target in one group.
- **Fixed windows don't.** They spread repeated content over 1.5–4.5 windows, and dilute almost as much as frame-level Shapley. On R3D-18, 8 DINOv2 groups give 1.07 per copy and 8 windows give 0.58. This answers "why not just use windows?"
- **The grouping must adapt to the sampling density.** A fixed number of groups works with copies, but not fully when a segment is sampled more densely (V-JEPA2: 0.79–0.89).
- **Ward linkage fails with a threshold** (V-JEPA2 frame-rate: 0.63). Its merge height grows with cluster size, so a longer run of similar frames is split into more groups. Average linkage doesn't have this problem.

## 9. Overall conclusions

1. **Frame-level attribution dilutes the credit of evidence shown in several frames.** Shapley (and so Play Fair) and leave-one-out do this on every model, and Integrated Gradients and occlusion on most. Duplication causes it, as the control clips show.
2. **Dilution matters in real videos.** Real clips are highly redundant (E2). When the more important part of a clip is spread over more frames, frame scores rank it wrongly more often (E3). Sampling a part more densely reduces each frame's credit, without changing the model's reliance on it (frame-rate test).
3. **Removal-based methods have a stronger form of the problem.** Leave-one-out and occlusion can make redundant evidence nearly invisible (E1, E2, frame-rate test).
4. **A second failure is separate from dilution.** Grad-CAM, and Shapley on most models, give repeated evidence too much credit in total.
5. **Giving credit to groups of frames removes dilution.** Group values stay close to their earlier values when evidence is repeated.
6. **The groups can be found without knowing which frames are redundant.** DINOv2 similarity with a fixed average-linkage threshold works in all 7 tests, with copies and with denser real frames.
7. **Fixed windows, a fixed number of groups, and Ward linkage don't fully work.** The grouping has to merge similar frames whatever their number.

## 10. Main limitations

- **Selection.** Only videos the model classifies correctly are used, and in E1 only those with an important frame.
- **Small samples on V-JEPA2:** 34 videos in E1, 68 in E2/E3 and 29 in the frame-rate test. Play Fair was run on only 2 models.
- **The yardsticks are model-based.** The I_t ratio and group Shapley values describe how the model behaves. They aren't value-free ground truth. E3 uses group Shapley as its yardstick, which is also the proposed remedy.
- **Group values don't follow a model that counts copies.** On MC3-18 the model relies on the duplicated frame 2.9 times as much, but its group value stays at about 1.
- **The practical groupings depend on τ,** one value per model, calibrated for a median of n/2 groups. We haven't tested how sensitive the results are to it. The threshold groupings also hit the 8-group cap in 41–73% of the longer MC3-18 and V-JEPA2 E1 clips.
- **The per-copy readout for adaptive groupings depends on group membership.** The target's group can contain different neighbouring frames in the reference and in the target clip. This explains part of the drop of `dino_k4` and `dino_thr` on V-JEPA2.
- **Top-1 loss rests on few cases** on R3D-50, MC3-18 and V-JEPA2.
- **One dataset per model:** UCF101, where appearance matters most, or SSv2, where motion matters most.

## 11. What remains for the paper

The comparison with fixed windows, and a way to choose the groups (the average-linkage threshold), are now done. What's left:
- **robustness checks:** how sensitive the results are to τ; E3 with a yardstick other than group Shapley (the circularity concern); and R3D with a removal that doesn't use neighbouring frames;
- **standard faithfulness measures,** such as deletion and insertion curves, for the grouping method;
- **a short theory section:** Shapley's symmetry rule forces interchangeable frames to share credit, and a group's value doesn't change when copies are added;
- **related work:** Play Fair, Shapley values with correlated features, Owen values, PartitionSHAP;
- **qualitative figures** of frame scores against group scores on real videos;
- optionally, the model-based grouping we deferred.
