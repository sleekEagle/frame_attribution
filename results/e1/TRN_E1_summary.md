# E1 summary: TRN on Something-Something v2

## In short

We tested what attribution methods do when an important frame appears several times in a clip, on TRN, a model that combines frames through "relation" modules.

TRN takes exactly 8 frames, so the only test possible is the target appearing 4 times instead of 2 (m = 4).

The model relies on the duplicated frame at least as much as before (1.32 times, in the pairs we used). But the attribution methods change:
- **Shapley** splits the credit exactly evenly between the copies. Each copy looks half as important as before.
- **Leave-one-out** gives each copy almost no credit (2% of before).
- **Occlusion** roughly splits the credit evenly.
- **Integrated Gradients** barely lowers the credit per copy, so the total grows to 1.68 times.
- **Grad-CAM** keeps the credit per copy, so the total nearly doubles.

In the control clips, the target's credit doesn't change. So duplication causes these changes. And no method matches the model on both the credit per copy and the total.

## 1. The question

When a frame the model relies on appears m times in a clip, does an attribution method still give it the right amount of credit?

"The right amount" is judged against the model itself. We measure how much the model relies on the frame before and after duplication (the I_t ratio). If the model relies on it as much as before, the frame's credit shouldn't change just because it's repeated.

## 2. How the test works

**Model.** `trn_official`: the original multi-scale TRN checkpoint from the TRN authors (`TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar`). It computes features for each frame with BN-Inception, then combines them with relation modules over 2 to 8 frames. It has 174 SSv2 classes, and it takes **exactly 8 frames**.

**Data.** `ssv2_sampled`: 1,044 SSv2 videos, 6 per class. We screened all of them.

**Clips.** We use the *reallocation* design. The clip always has 8 slots:
- We pick 4 frames from the video (every other one of 8 sampled frames). We call them *contents*.
- **Reference clip:** each content appears twice, in two neighbouring slots (m = 2).
- **Target clip:** the target t gets 4 slots. To make room, the two other contents (the *losers*) drop from two copies to one.
- **Control clip:** the same losers drop to one copy, but the freed slots go to the least important content (the *recipient*). The target keeps its two copies.

So at m = 4, the target clip has t four times, the recipient twice, and the two other contents once each. **m = 4 is the only level the 8 slots allow.** Adding copies (making the clip longer) isn't possible, because the relation modules need exactly 8 frames.

Each copy is either exact, or has a little random noise added (σ = 2/255).

**Removing frames.** A removed slot is filled with the nearest remaining frame before it, and, in a second clip, with the nearest one after it. The model's two outputs are averaged (`late` freeze). So every clip has 8 frames.

**Fixed randomness.** TRN's relation modules pick random subsets of frames. Our code fixes that randomness, so the same clip always gives the same output.

**Choosing the target.** We measure how much the model needs each content: we remove every copy of it and see how much the predicted class's probability drops. This is the content's *importance* I. One target is picked from the top 2 contents, and a second from ranks 3–4 when available. Both need I > 0.01. No attribution method is used to choose them.

**The per-pair gate.** A target clip and its control are used only if both give the reference prediction.

**Methods:**
- Shapley (freeze), computed **exactly** over the 8 slots, so one seed is enough;
- leave-one-out (freeze);
- occlusion (one frame replaced by a blank frame);
- Integrated Gradients (32 steps, blank baseline);
- Grad-CAM on the per-frame features.

## 3. Which videos were used

| step | number |
|---|---|
| videos screened | 1,044 |
| left out: the model got them wrong | 778 |
| left out: no single content mattered enough (I ≤ 0.01) | 21 |
| **kept** | **245 videos, 375 targets** (245 top, 130 middle) |
| target–control pairs that passed the gate | **202 of 375 (54%)** with exact copies; 200 with noisy copies |
| videos with at least one passing pair | 166 (exact), 165 (noisy) |
| passing pairs, by target | top 158 of 245 (64%), middle 44 of 130 (34%) |

The model gets only about 30% of these videos right: 30.0% in a 60-video check, and 33.6% on the full SSv2 test folder. That's why most videos are left out.

## 4. Does duplication change the model?

All 375 targets, exact copies. Noisy copies are within 0.5 percentage points.

| clip | same prediction as the reference | change in the predicted class's probability | how much the model relies on the target (I_t ratio) |
|---|---|---|---|
| original 8 frames vs the reference clip | 87.8% | −0.047 | – |
| target clip, m = 4 | **72.3%** | −0.142 | 1.00 |
| control clip, m = 4 | **60.0%** | −0.241 | 0.58 |

What this shows:
- **Showing each content twice already changes the prediction in 12% of videos** (6% for R3D-50). TRN combines frames in order through its relation modules, so showing 4 contents in pairs, instead of 8 different frames, affects it more.
- **At m = 4 the clip changes a lot.** Half of the other content drops to a single slot. In the control, the least important content fills 4 of the 8 slots. That's why the control changes the prediction even more often than the target clip.
- **The model relies on the target as much as before** (I_t ratio 1.00 over all targets). In the pairs used for attribution, it's **1.32** in the target clip and 0.98 in the control.
- **With only one level of m, we can't drop the harder levels.** So we use the per-pair gate, as for R3D, which keeps 54% of pairs.

## 5. Results

### 5.1 Credit per copy and credit in total

- **per-copy ratio:** the credit of one copy, divided by its credit in the reference. If the credit is split evenly between the copies, it's 2/m = **0.50**.
- **total ratio:** the credit of all copies added up, divided by the total in the reference. It stays at 1 if the credit is split evenly, and becomes m/2 = 2 if each copy keeps its full credit.
- **best copy:** the credit of the copy with the highest credit.
- **β:** with only one level, β is just the per-copy ratio on a log scale: β = log(per-copy ratio) / log(2). β = −1 means 0.50 (split evenly), and β = 0 means 1.0 (each copy keeps its credit).

Exact copies, medians over the 202 pairs used. Noisy copies are within 0.03 (per copy) and 0.06 (β).

| method | per-copy ratio | best copy | total ratio | β, target [95% range] | control per-copy ratio | β, control |
|---|---|---|---|---|---|---|
| (credit split evenly) | 0.50 | | 1.00 | −1 | | |
| **(how much the model relies on t)** | | | **1.32** | | | |
| Shapley | **0.50** | 0.56 | **0.99** | −0.94 [−1.11, −0.81] | 0.98 | −0.03 |
| Leave-one-out | **0.02** | 0.08 | 0.04 | −2.60 [−3.19, −2.25] | 1.03 | +0.25 |
| Occlusion | **0.55** | 0.97 | 1.11 | −0.75 [−0.94, −0.52] | 1.09 | +0.17 |
| Integrated Gradients | 0.84 | 1.05 | 1.68 | −0.21 [−0.28, −0.16] | 0.97 | −0.02 |
| Grad-CAM | 0.96 | 0.96 | **1.92** | −0.05 [−0.07, −0.02] | 1.00 | −0.00 |

Notes:
- β only uses targets whose reference credit is positive: Shapley 200, Integrated Gradients 192, Grad-CAM 182, leave-one-out 180, occlusion 170.
- For leave-one-out, β below −1 just means the credit went to almost 0. The per-copy ratio describes it better.

### 5.2 What each method does

- **Shapley splits the credit exactly evenly.** Each copy gets 0.50 of its reference credit, and the total stays at 0.99. So each copy looks half as important as before, while the model relies on the target slightly more (1.32).
- **Leave-one-out gives each copy almost no credit** (2% of the reference). Removing one copy leaves the others in the clip, so the output barely changes. A content the model needs is reported as unimportant.
- **Occlusion roughly splits the credit evenly** (0.55 per copy, total 1.11). The control doesn't change in the same way.
- **Integrated Gradients barely lowers the credit per copy** (0.84). So the total grows to 1.68 times the reference, more than the model's 1.32.
- **Grad-CAM keeps the credit per copy** (0.96), so the total nearly doubles (1.92).

In the control clips, the target's credit stays the same for every method (0.97 to 1.09). So the changes above are caused by duplicating the target.

### 5.3 Does duplication change the ranking of frames?

- **Inversion:** a content ranked clearly below the target in the reference is ranked above it after duplication, while the model still ranks the target higher.
- **Top-1 loss:** the target was the method's top content in the reference, but isn't any more. All methods here give the same result every time, so random noise adds nothing.

Exact copies:

| method | inversions, target | inversions, control | top-1 loss, target (number of cases) | top-1 loss, control |
|---|---|---|---|---|
| Shapley | **36.4%** | 7.8% | **0.61** (105) | 0.31 |
| Leave-one-out | **39.0%** | 17.4% | **0.83** (102) | 0.57 |
| Occlusion | 16.7% | 14.6% | 0.43 (69) | 0.19 |
| Integrated Gradients | 9.5% | 10.4% | 0.19 (69) | 0.16 |
| Grad-CAM | 14.6% | 5.7% | 0.25 (68) | 0.16 |

The control's top-1 loss isn't 0, because the control clip also differs from the reference: the losers drop to one copy, and the recipient gets more.

What this shows:
- **Duplication pushes the target down Shapley's ranking.** Inversions rise to 36% (8% in the control), and the target loses Shapley's top rank in 61% of cases.
- **Leave-one-out pushes it down the most** (39% inversions, top-1 loss 0.83).

## 6. Conclusions

1. **Duplication changes a content's credit, although the model relies on it at least as much as before.** In the pairs used, the model relies on the target 1.32 times as much as in the reference. Yet Shapley, leave-one-out and occlusion give each copy far less credit. In the control clips, the target's credit doesn't change (0.97–1.09). So duplication causes the change.

2. **Each method changes the credit in its own way.**
   - Shapley: splits it exactly evenly, so each copy looks half as important. Duplication also pushes the target down its ranking.
   - Leave-one-out: almost no credit per copy. A content the model needs looks unimportant.
   - Occlusion: roughly splits it evenly.
   - Integrated Gradients: little change per copy, so too much in total (1.68 against 1.32).
   - Grad-CAM: no change per copy, so the total nearly doubles.

3. **No method matches the model on both measures.** Either the credit per copy falls far below the reference (Shapley, leave-one-out, occlusion), or the total is higher than the model's reliance (Integrated Gradients, Grad-CAM). So a content's credit depends on how often it appears, not only on how much the model uses it.

4. **How a method fails depends on the model.** Grad-CAM inflates the total on every model we tested, and leave-one-out collapses on every model except V-JEPA2. But Shapley splits the credit exactly evenly on TRN, and only partly on R3D-50 (0.76 per copy at m = 4). Integrated Gradients does the opposite: close to an even split on R3D-50 (0.54), little change on TRN (0.84). So a method's behaviour under duplication can't be judged from one model. `results/E1_report.md` compares all five models.

5. **Noisy copies give the same results as exact copies** (per-copy ratios within 0.03).

## 7. Limitations

- **Only one level (m = 4).** The 8 slots allow no other level, so the results are single ratios, not trends across several m.
- **Strong selection.**
  - Only 245 of 1,044 videos are classified correctly and have an important content.
  - Only 54% of their target–control pairs keep the reference prediction in both clips. Middle-ranked targets pass less often (34%) than top-ranked ones (64%).
  - So the pairs used are those whose prediction survives a large change to the clip.
- **The reference isn't the original clip.** It shows 4 contents, each in two neighbouring slots. It changes TRN's prediction in 12% of videos.
- **The control changes the clip a lot** (60% keep the prediction): the least important content fills half the slots.
- **Only the official TRN.** Play Fair's own TRN can't be tested: its files (`trn.pth`, `trn_8_frames.pth`) were deleted from the authors' Dropbox, and our local copies are just "File Deleted" web pages.
- **No Play Fair or methods that delete frames.** Our code needs exactly 8 frames for this model.

## 8. Still to do (optional)

1. **Play Fair on the official TRN.** Play Fair scores a subset of k frames with the relation module for k frames, and this model has modules for 2 to 8 frames. This needs a new way to evaluate subsets in our code. It would also need a check of the model's accuracy for each k, like the clip-length check for R3D, because the scores would then depend on the smaller modules.
2. **Play Fair's own TRN**, if copies of `trn.pth` and `trn_8_frames.pth` can be found (for example from the Play Fair authors). Put them in `play-fair/checkpoints/backbones/` and `play-fair/checkpoints/features/`; the commands below then work with `--model trn`.

## 9. Files and how to reproduce

Files in `results/e1/`:

| file | what it contains |
|---|---|
| `trn_official_conditions.jsonl`, `trn_official_skipped.jsonl` | the clips built for the 245 kept videos, and the videos left out |
| `trn_official_gate.csv` | the gate check (section 4) |
| `trn_official_attributions.jsonl` | the per-frame scores for every clip and method |
| `trn_official_rows.csv`, `trn_official_slopes.csv`, `trn_official_summary.csv`, `trn_official_beta.csv` | the measures, and β with 95% ranges |
| `trn_official_build.log`, `trn_official_attribute.log` | run logs |

Commands (from the repo root, in `tc_env`). Every step resumes from existing output, so to start fresh, move the old `results/e1/trn_official_*` files away first.
```
python motivation/e1_build_conditions.py --model trn_official      # all 1,044 videos (~6 min)
python motivation/e1_metrics.py gate --model trn_official
python motivation/e1_attribute.py --model trn_official --seeds 0   # ~2 h
python motivation/e1_metrics.py attr --model trn_official
```
This needs `weights_only=False` in `models/trn_official.py` (PyTorch 2.6 or later) and the TRN checkpoint in `play-fair/checkpoints/backbones/`.
