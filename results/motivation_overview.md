# Attribution dilution in video: overview of E1, E2, E3 and the group follow-ups

This report brings together everything we have tested about attribution dilution so far. It gives the question, the method and the main numbers for each experiment, and links to the detailed reports.

| detailed report | covers |
|---|---|
| `results/E1_report.md` and `results/e1/*_E1_summary.md` | E1, controlled duplication, on all models and per model |
| `results/design_choice_report.md` | which models can be used, and why |
| `results/e23/E23_summary.md` | E2 (redundancy in real clips) and E3 (the dilution test in real clips) |
| `results/group_attribution_summary.md` | group values on the E1 clips, and the frame-rate test |

## In short

1. **Dilution is real, and duplication causes it** (E1). When an important frame is shown several times, the model relies on it about as much as before (0.87–1.32 times, on four of five models). But every attribution method gives each copy less credit: 0.16–0.62 of the frame's earlier credit for Shapley, and about 0 for leave-one-out.
2. **Real clips are highly redundant** (E2). Removing frames one at a time reveals almost nothing: the single-frame importances add up to about 0% of what the clip contributes. Removing the 4 top-ranked frames still leaves 75–95% of it.
3. **Dilution misleads users in real videos** (E3). When the more important part of a clip is spread over many frames, Shapley's frame scores rank it below a less important, shorter part significantly more often. On V-JEPA2: 42% of the time, against 25% for parts of the same size.
4. **It happens with denser real frames too** (frame-rate test). On V-JEPA2, sampling the most important quarter of a clip twice as densely leaves the model's reliance on it unchanged (0.96). But each of its frames gets 0.67 of its earlier Shapley credit.
5. **Giving credit to groups removes it** (group follow-up). Shapley with all copies of a frame as one player keeps the group's credit at 0.94–1.12 of its earlier value on all five models. DINOv2 clustering finds those groups in 86–100% of clips.

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
| R3D-50 | UCF101 | reallocation | fill with a neighbouring frame (freeze) | E1, E2/E3, group |
| R3D-18 | UCF101 | reallocation | freeze | E1, E2/E3, group |
| MC3-18 | UCF101 | insertion | delete | E1, E2/E3, group, frame-rate |
| V-JEPA2 | SSv2 | insertion | delete | E1, E2/E3, group, frame-rate |
| TRN | SSv2 | reallocation (8 frames) | freeze | E1, group |

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

**Main numbers** (Shapley, exact copies, highest m):

| | R3D-50 (×4) | R3D-18 (×4) | MC3-18 (×9) | V-JEPA2 (×9) | TRN (×2) |
|---|---|---|---|---|---|
| model's reliance (I_t ratio) | 1.28 | 1.30 | 3.22 | 0.87 | 1.32 |
| Shapley, credit of one copy | 0.62 | 0.52 | 0.48 | 0.16 | 0.50 |
| Shapley, all copies added up | 2.47 | 2.07 | 4.33 | 1.45 | 0.99 |
| β, target (control) | −0.37 (−0.01) | −0.47 (+0.03) | −0.31 (−0.12) | −0.71 (−0.02) | −0.94 (−0.03) |
| top-1 loss, target vs noise alone | 0.71 vs 0.47 | 0.75 vs 0.38 | 0.24 vs 0.62 | too few cases | 0.61 vs 0 |

**What it shows:**
- **Every method dilutes, and duplication causes it.** For Shapley and leave-one-out, the target's β is clearly negative on all five models, and the control's is near 0.
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

## 8. Overall conclusions

1. **Frame-level attribution dilutes the credit of evidence shown in several frames.** This happens on every model and every method we tested. Duplication causes it, as the control clips show.
2. **Dilution matters in real videos.** Real clips are highly redundant (E2). When the more important part of a clip is spread over more frames, frame scores rank it wrongly more often (E3). Sampling a part more densely reduces each frame's credit, without changing the model's reliance on it (frame-rate test).
3. **Removal-based methods have a stronger form of the problem.** Leave-one-out and occlusion can make redundant evidence nearly invisible (E1, E2, frame-rate test).
4. **A second failure is separate from dilution.** Grad-CAM, and Shapley on most models, give repeated evidence too much credit in total.
5. **Giving credit to groups of frames removes dilution.** Group values stay close to their earlier values when evidence is repeated, and DINOv2 can supply the groups.

## 9. Main limitations

- **Selection.** Only videos the model classifies correctly are used, and in E1 only those with an important frame.
- **Small samples on V-JEPA2:** 34 videos in E1, 68 in E2/E3 and 29 in the frame-rate test. Play Fair was run on only 2 models.
- **The yardsticks are model-based.** The I_t ratio and group Shapley values describe how the model behaves. They aren't value-free ground truth. E3 uses group Shapley as its yardstick, which is also the proposed remedy.
- **The grouping is still open.** In the group follow-up, DINOv2 used the true number of groups, which isn't known for real videos. At 16 frames, DINOv2 groups are no more redundant than fixed windows (E2/E3), so we haven't yet shown that similarity-based groups beat simple windows.
- **One dataset per model:** UCF101, where appearance matters most, or SSv2, where motion matters most.

## 10. What remains for the paper

These are not further E1–E3 experiments; they belong to the method part:
- evaluate the grouping method against frame-level methods and fixed windows, using the E3 dilution test and standard deletion and insertion curves;
- decide how to choose the number of groups, or use a model-based grouping (an idea we've deferred);
- a short theory section: Shapley's symmetry rule forces interchangeable frames to share credit, and a group is unchanged by adding copies;
- qualitative figures of frame scores against group scores on real videos.
