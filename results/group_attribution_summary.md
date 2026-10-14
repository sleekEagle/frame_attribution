# Group attribution and the frame-rate test

This summary covers two follow-ups to E1 and E2/E3:
- **Step 2, group values on the E1 clips.** If we give credit to groups of frames instead of single frames, does dilution go away?
- **Step 1, the frame-rate test.** Does dilution also happen when the extra frames are real neighbouring frames, not copies?

E1 is described in `results/E1_report.md`, and E2/E3 in `results/e23/E23_summary.md`.

## In short

**Group values remove dilution on all five models.** When a frame is duplicated, frame-level Shapley gives each copy 0.14–0.61 of the frame's earlier credit. Shapley with all the copies as one group gives the group 0.94–1.12 of its earlier credit. The model's own reliance on the frame stays at 0.87–1.32 on four of the models.

**Group values keep the ranking.** On R3D-18 and TRN, frame scores move the duplicated frame off the top in 61–64% of cases. Group values do this in 0–4% of cases.

**DINOv2 finds the groups.** With the same number of groups as the true grouping, DINOv2 puts the copies of a frame together in 86–100% of clips. Its group values are then the same as with the true grouping.

**Dilution also happens with real frames.** On V-JEPA2, we sampled the most important quarter of a clip twice as densely. The model relied on that quarter as before (0.96), and so did its group value (1.00). But each frame's Shapley credit fell to 0.67 of its earlier value. This also holds on the 25 clips where every added frame is a real, distinct frame.

**One limitation.** Group values don't rise when the model itself counts copies. On MC3-18, the model relies on a frame shown 9 times about 2.9 times as much, but its group value stays at about 1.0.

## 1. Words used

- **Content:** one distinct frame of the video. In E1, a content can be shown several times in a clip (its *copies*).
- **Importance I:** how much the model relies on something. It is the drop in the predicted class's probability when that thing is removed from the clip (every copy of a content, or every frame of a segment).
- **I_t ratio:** the target's importance in a changed clip, divided by its importance in the reference clip. It is the model's own answer to "does the model rely on the target more or less than before?"
- **Frame Shapley:** Shapley values with each frame as a player. Copies are separate players, so they share credit.
- **Group Shapley:** Shapley values with each group of frames as one player. A group is added or removed as a whole, so its credit is never divided among its frames.
- **Ratio:** a readout in the changed clip divided by the same readout in the reference clip. A ratio of 1 means no change.
- **β (dilution slope):** how fast a ratio falls as the number of copies grows. It is the slope of log(ratio) against log(copies relative to the reference). β = 0 means no change. β = −1 means the credit falls in proportion, as when it's split evenly among the copies.
- **Top-1 loss:** among targets that ranked first in the reference clip, the share that no longer rank first in the changed clip.

## 2. Step 2: group values on the E1 clips

### 2.1 How it works

We reused E1's clips and its frame-level Shapley values (`motivation/e1_group.py`). For each clip that passed E1's gate, we computed group Shapley with these groupings:

| grouping | groups | used for |
|---|---|---|
| **oracle** | one group per content: all copies of a content form one group. We know the copies, because we built the clips. | R3D-50, R3D-18, MC3-18, TRN |
| **dino** | the clip's frames clustered with DINOv2. Only neighbouring frames can be joined. Cut to the same number of groups as the oracle. | all five models |
| **dino_half** | the same DINOv2 tree, cut to half as many groups. Groups are coarser, so the target can share a group with other frames. | R3D-50, R3D-18, MC3-18, TRN |
| **pairs** | all copies of the target form one group. The other contents are joined into neighbouring pairs, never across the target. | V-JEPA2 |

**Why V-JEPA2 needs the pairs grouping.** V-JEPA2 uses E1's insertion design. Its clips have 16 contents, so the oracle grouping has 16 groups. Exact Shapley over 16 groups needs 2^16 = 65,536 model evaluations per clip, which is too many. A first run used 32 random orderings instead, and the values were too noisy to read: at m = 9, the middle half of the ratios ran from −0.31 to 1.46. The pairs grouping has 9 groups, so exact Shapley needs only 512 evaluations, and has no sampling noise. The reference clip is grouped the same way as the clip it's compared with.

Everything else is as in E1:
- the same predicted class;
- the same removal (freeze on R3D-50, R3D-18 and TRN; deletion on MC3-18 and V-JEPA2);
- the same prior.

All readouts compare the target's group with the same group in the reference clip.

| model | videos | design | copies of the target (reference) |
|---|---|---|---|
| R3D-50 | 60 | reallocation | 4, 6, 8 (2) |
| R3D-18 | 60 | reallocation | 4, 6, 8 (2) |
| TRN | 245 | reallocation | 4 (2) |
| MC3-18 | 60 | insertion | 3, 5, 9 (1) |
| V-JEPA2 | 34 (32 with pairs) | insertion | 3, 5, 9 (1) |

Exact copies only. For V-JEPA2, two videos had no exact-copy clips that passed the gate.

### 2.2 Results

Target clips, exact copies, at the highest number of copies. Medians over targets:

| | R3D-50 (×4) | R3D-18 (×4) | TRN (×2) | MC3-18 (×9) | V-JEPA2 (×9) |
|---|---|---|---|---|---|
| model's reliance (I_t ratio) | 1.01 | 1.27 | 1.32 | 2.88 | 0.87 |
| frame Shapley, one copy | 0.61 | 0.52 | 0.49 | 0.54 | 0.14 |
| frame Shapley, all copies added up | 2.45 | 2.09 | 0.99 | 4.86 | 1.30 |
| **group Shapley** (oracle; pairs on V-JEPA2) | **1.09** | **1.06** | **1.12** | **1.02** | **0.94** |
| group Shapley, DINOv2 | 1.09 | 1.06 | 1.12 | 1.02 | – |
| group Shapley, DINOv2 with half as many groups | 0.98 | 0.94 | 1.00 | 1.01 | – |
| β, frame Shapley per copy | −0.35 | −0.46 | −0.94 | −0.24 | −0.64 |
| **β, group Shapley** | **+0.06** | **+0.05** | **+0.17** | **+0.09** | **+0.04** |

"×4" means 4 times as many copies as in the reference. The 95% ranges of the group β are [+0.02, +0.10], [+0.01, +0.07], [+0.12, +0.21], [+0.01, +0.17] and [−0.04, +0.06]. Those of the frame β all end at −0.18 or lower.

R3D-50's I_t ratio here (1.01) differs from the E1 report (1.28), because this run uses 60 of E1's 120 videos.

**In the control clips,** the extra copies go to other contents. There the target's group value stays at 0.94–1.04 of the reference on every model. So the group value changes only when the target itself is copied, and then only a little.

**On V-JEPA2, the exact values follow the model.** Across targets, the group value's rank correlation with the I_t ratio is 0.72 at m = 9, against 0.53 for frame Shapley per copy. In one video, the model stops relying on the target once it's copied 9 times (I_t ratio 0.08 and −1.82 for its two targets). The group value drops with it (−0.75 and −0.86).

**Ranking.** Top-1 loss at the highest number of copies:

| | R3D-50 | R3D-18 | TRN |
|---|---|---|---|
| frame Shapley, target clip | 0.50 | 0.64 | 0.61 |
| frame Shapley, control clip | 0.61 | 0.56 | 0.31 |
| group Shapley, target clip | 0.19 | 0.00 | 0.04 |
| group Shapley, control clip | 0.38 | 0.12 | 0.08 |
| targets ranked first in the reference (frame / group) | 18 / 26 | 25 / 24 | 105 / 100 |

- **On R3D-18 and TRN, frame scores move the duplicated target off the top, and group values don't.**
- **On R3D-50, the frame scores change the top rank as often in the control clips** as in the target clips, so the change isn't caused by duplication. The group values change it less often in both.
- MC3-18 and V-JEPA2 have too few top-ranked targets (17, and 3–8) to read.

**DINOv2.**
- At the same number of groups, DINOv2 puts the target's copies in a group of their own in 100% (R3D-50), 99% (R3D-18, TRN), 97% (MC3-18) and 86% (V-JEPA2) of clips. On the first four models, its group values are then the same as the oracle's.
- With half as many groups, it does so in only 24–45% of clips: the copies usually share a group with neighbouring frames. Those groups still aren't diluted (ratios 0.94–1.01). But they lose the top rank more often (top-1 loss 0.18–0.39), because a mixed group's value isn't the target's alone.

### 2.3 What this shows

1. **Group values remove dilution.** On every model, frame Shapley's credit per copy falls as copies are added (β −0.24 to −0.94). The group value stays close to its reference value (β +0.04 to +0.17). This is what the theory predicts: Shapley's symmetry rule makes interchangeable frames share credit, and a group is a single player, so it has nothing to share.
2. **Group values keep the ranking** on the models where we can measure it.
3. **The grouping doesn't have to be given.** DINOv2 finds the copies in almost every clip when it uses the right number of groups.
4. **Group values stay flat when the model counts copies.** On MC3-18 the model relies on the target 2.9 times as much at m = 9. The group value stays at 1.02, and frame Shapley's total overshoots (4.86). A group value measures the content's value as a unit, whatever its number of copies. That is what the dilution definition asks for. But it means group values don't follow the model when the model itself gives weight to repetition.

## 3. Step 1: the frame-rate test

### 3.1 How it works

In E1 we made copies ourselves. Here, the extra frames are real frames from the same stretch of the video (`motivation/fr_run.py`):

1. **Base clip:** the model's own 16 frames, split into 4 segments of 4 frames each, in time order.
2. **Target and control:** exact group Shapley over the 4 segments. The **target segment** is the most important one (group value above 0.01). The **control segment** is the least important one.
3. **Denser clips:** the target segment's 4 frames are replaced by 4d frames, sampled evenly over the same stretch of the video (d = 2 or 3). The control clip densifies the control segment in the same way. The clip gets longer, so only models that accept other lengths can be used: MC3-18 and V-JEPA2.
4. **For each clip:** the model's prediction; the frame-level methods; group Shapley over the 4 segments; and each segment's importance I.

Readouts are for the target segment, relative to the base clip:
- the model's reliance on it (I ratio);
- its group Shapley value (group ratio);
- its frames' average score (per frame).

Clips whose prediction differs from the base clip are left out.

| model | data | videos screened | videos used | densified clips with the base prediction |
|---|---|---|---|---|
| V-JEPA2 | SSv2 | 40 | 29 (11 misclassified) | 113 of 116 |
| MC3-18 | UCF101 | 60 | 49 (11 misclassified) | 190 of 196 |

**Repeated frames.** Some SSv2 videos are short. Then sampling 3 times as densely runs out of distinct frames, and some frames appear twice. This happens in 47 of V-JEPA2's 116 densified clips, mostly at d = 3. At d = 2, 25 of the 29 target clips use only distinct frames. On MC3-18, it happens in 1 clip of 196.

### 3.2 Results: V-JEPA2

Target segment, medians:

| readout | d = 2 | d = 3 | β [95% range] | control segment, β |
|---|---|---|---|---|
| model's reliance (I ratio) | 0.96 | 0.94 | −0.01 [−0.08, +0.10] | +0.02 |
| group Shapley | 1.00 | 0.98 | −0.01 [−0.10, +0.01] | −0.01 |
| **frame Shapley, per frame** | **0.67** | **0.49** | **−0.50 [−0.84, −0.29]** | +0.06 [−0.04, +0.18] |
| frame Shapley, all frames added up | 1.35 | 1.47 | +0.50 | +0.06 |
| occlusion, per frame | 0.04 | 0.04 | −0.95 | +0.04 |
| Grad-CAM, per frame | 0.80 | 0.69 | −0.34 | −0.23 |

**With distinct frames only** (d = 2, 25 clips), frame Shapley per frame is 0.67, the same as over all clips. The model's reliance is 0.95, and the group value 1.00. So the effect isn't caused by repeated frames.

**Ranking with frame Shapley.** Inversions are other segments with a lower group value than the target that get a higher average frame score:

| | base clip | d = 2 | d = 3 |
|---|---|---|---|
| target segment densified | 0.26 | 0.23 | **0.41** |
| control segment densified | 0.26 | 0.07 | 0.12 |

Top-1 loss at d = 3: frame scores 0.71 (14 cases), group values 0.29 (28 cases). In the control clips: 0.21 and 0.07.

What this shows:
- **Dilution happens with real frames.** When the most important segment is sampled more densely, the model relies on it as before, and its group value stays the same. But each of its frames gets less Shapley credit: 0.67 at d = 2 and 0.49 at d = 3. Densifying the control segment doesn't change the target's credit, so the drop is caused by the denser sampling.
- **At d = 3, frame scores rank the segment wrongly more often:** 41% of the comparisons, against 26% in the base clip.
- **Occlusion's per-frame credit drops to almost 0.** Blanking one frame changes little when its neighbours show nearly the same thing. This matches E1 and E2.
- **Grad-CAM's credit per frame also falls in the control**, so for Grad-CAM the drop isn't specific to the denser segment.

### 3.3 Results: MC3-18

| readout (target segment) | d = 2 | d = 3 | β | control segment, β |
|---|---|---|---|---|
| model's reliance (I ratio) | 0.90 | 0.90 | +0.32 | +0.03 |
| group Shapley | 0.95 | 0.93 | −0.05 | +0.01 |
| frame Shapley, per frame | 0.73 | 0.62 | −0.44 | −0.20 |

- **Frame credit per frame falls, but only partly because of the denser sampling.** The target's credit also falls when the control segment is densified (β −0.20), because the clip gets longer. The part caused by densifying the target is about −0.24.
- **The ranking doesn't get worse.** MC3-18's 4 segments have almost the same group value: the most important leads the next by about 7% of its value. So there's no clearly more important segment to lose its rank. That is the same reason MC3-18 showed no effect in E3.
- The target segment's group value changes the top rank in 52–67% of cases, for the same reason: the lead is so small that a 5% change reverses it.

### 3.4 What this shows

1. **Dilution happens with real, distinct frames,** not only with copies. On V-JEPA2, a segment sampled twice as densely keeps the model's reliance and its group value. But each of its frames gets about two thirds of its earlier Shapley credit.
2. **Group values over segments don't change** with the sampling density (0.93–1.00 on both models).
3. **Where all parts of a clip matter about equally (MC3-18 on UCF101),** there's no ranking for dilution to change.

## 4. Conclusions

1. **Giving credit to groups removes dilution.** With copies (step 2, five models) and with denser real frames (step 1, V-JEPA2), the group value stays close to its earlier value, while frame-level Shapley's credit per frame falls.
2. **Group values keep the ranking** where frame scores lose it: on R3D-18 and TRN in step 2, and on V-JEPA2 in step 1.
3. **DINOv2 can supply the groups.** At the right number of groups, it finds the copies in 86–100% of clips. Coarser groups stay undiluted but mix the target with other frames.
4. **Group values measure a content as a unit, whatever its number of frames.** When a model itself gives more weight to repeated frames (MC3-18), group values don't follow that.

## 5. Limitations

- **The grouping depends on the model's input design.** V-JEPA2 uses the pairs grouping, so it can't be compared with the other models' oracle numbers one to one. Its reference uses the same grouping, so its own ratios are fair.
- **Small samples:** V-JEPA2 has 32 videos in step 2 and 29 in step 1. Top-1 loss can't be read on MC3-18 and V-JEPA2.
- **Repeated frames at d = 3 on V-JEPA2.** Most d = 3 clips repeat some frames. The d = 2 result, from distinct frames, is the clean one.
- **Exact copies only** in step 2. E1 showed that noisy copies give the same frame-level results.
- **Only two models in step 1,** because the others need a fixed clip length.
- **DINOv2 with the matched number of groups** uses the oracle's group count, which isn't known for real videos. Choosing the number of groups is part of the method, and is still open.

## 6. Files and how to reproduce

| file | what it contains |
|---|---|
| `results/e1/<model>_group_attributions.jsonl` | group values per clip and grouping (`trn_official`, `r3d`, `r3d_18`, `mc3_18`, `vjepa2`) |
| `results/e1/<model>_group_rows.csv`, `_group_summary.csv`, `_group_metrics.log` | step 2 readouts |
| `results/fr/<model>_fr.jsonl`, `_fr_skipped.jsonl` | step 1 data (`vjepa2`, `mc3_18`) |
| `results/fr/<model>_fr_rows.csv`, `_fr_summary.csv`, `_fr_metrics.log` | step 1 readouts |
| `results/kaggle_logs/` | run logs |

Commands (all runs except TRN were on Kaggle, 2 × T4, with `--shard` / `--num-shards 2`):
```
# step 2
python motivation/e1_group.py --model trn_official --fp16
python motivation/e1_group.py --model r3d_18 --fp16 --limit 60
python motivation/e1_group.py --model r3d    --fp16 --limit 60
python motivation/e1_group.py --model mc3_18 --fp16 --limit 60
python motivation/e1_group.py --model vjepa2 --fp16 --dup-types exact --groupings pairs
python motivation/e1_group_metrics.py --model <model>

# step 1
python motivation/fr_run.py --model vjepa2 --dataset ssv2 --limit 40 --fp16 --n-perm 32 --methods shapley_drop loo_drop occlusion gradcam
python motivation/fr_run.py --model mc3_18 --limit 60 --fp16
python motivation/fr_metrics.py --model <model>

# tests
python motivation/test_e23.py
```
Run the metrics in an environment without the MKL crash (for example `torch_plot`).
