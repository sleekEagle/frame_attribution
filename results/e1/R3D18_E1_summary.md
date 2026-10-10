# E1 summary: R3D-18 on UCF101

## In short

R3D-18 is a second, smaller 3D ResNet. We ran it to check whether the R3D-50 results repeat on a different network from a different source. They do.

The model barely changes when a frame is duplicated. It keeps its prediction in about 95% of clips, and it relies on the duplicated frame about as much as before (1.06 to 1.30 times).

The attribution methods do change:
- **Leave-one-out** gives each copy almost no credit.
- **Shapley** gives each copy less credit, but the copies more credit in total.
- **Integrated Gradients** gives each copy less credit. Its total grows a little.
- **Occlusion** gives each copy less credit, and its total stays close to the model's.
- **Grad-CAM** gives the copies more credit in total.

In the control clips, the target's credit doesn't change for any method. So the changes come from duplicating the target.

## 1. The question

When a frame the model relies on appears m times in a clip, does an attribution method still give it the right amount of credit?

"The right amount" is judged against the model itself, as for R3D-50. We measure how much the model relies on the frame before and after duplication (the I_t ratio). If the model's reliance doesn't change, the frame's credit shouldn't change just because it's repeated.

## 2. How the test works

The test is the same as for R3D-50 (see `R3D_E1_summary.md`, section 2). In brief:

**Model.** R3D-18 from torchvision, pretrained on Kinetics-400 and fine-tuned on UCF101 split 1, published as `dronefreak/r3d-18-ucf101` (wrapper: `models/torchvision_ucf101.py`). It takes 16 frames at 112 px. Its test accuracy is 82.95% on the full split-1 test set (reported: 83.43%).

**Why reallocation.** R3D-18's accuracy changes in steps with clip length (at 4/5, 8/9 and 16/17 frames). The same frames shown at length 16 are up to 30 points more accurate than as a short clip. So clips must stay at 16 frames, and removed frames are filled by freezing, not deleted. See `results/design_choice_report.md`, section 6.

**Clips.** 8 contents (every other one of 16 frames).
- **Reference:** each content twice (m = 2).
- **Target clip:** the target gets m slots, taken from other contents (the *losers*), which drop to one copy.
- **Control clip:** the same losers drop to one copy, and the freed slots go to the least important content (the *recipient*).

We test m = 4, 6 and 8, with exact copies and with slightly noisy copies (σ = 2/255).

**Targets.** Chosen by the model's own importance I(c), never by an attribution method. One "high" target from the top 2 contents, and one "mid" target from ranks 3–4 when available. Both need I > 0.01.

**Methods.** Shapley (freeze; 64 orderings, 2 seeds), leave-one-out (freeze), occlusion, Integrated Gradients (32 steps), Grad-CAM (`layer2`). Methods that delete frames aren't used, because of the length effect above.

## 3. Which videos were used

| step | number |
|---|---|
| test videos screened | 500 |
| left out: the model got them wrong | 95 |
| left out: no single content mattered enough (I ≤ 0.01) | 60 |
| kept | 345 videos, 547 targets |
| **used for attribution** (the first 120 kept videos) | **120 videos, 197 targets** |
| target–control pairs that passed the gate | 189 at m = 4, 190 at m = 6, 186 at m = 8 |

We attributed 120 videos to match R3D-50.

## 4. Does duplication change the model?

Checked on all 345 kept videos (547 targets), before running any attribution method. Exact copies; noisy copies are within 0.5 percentage points.

| clip | same prediction as the reference | change in the predicted class's probability | how much the model relies on the target (I_t ratio) |
|---|---|---|---|
| original 16 frames vs the reference clip | 95.7% | +0.004 | – |
| target clip, m = 4 / 6 / 8 | 96.7% / 96.2% / 94.9% | +0.002 / +0.008 / −0.002 | 1.02 / 1.17 / 1.33 |
| control clip, m = 4 / 6 / 8 | 96.7% / 96.9% / 94.5% | −0.006 / −0.002 / −0.009 | 0.95 / 0.91 / 0.86 |

What this shows:
- **The model relies on the duplicated target about as much as before**, a little more at high m (1.02 to 1.33).
- **The model keeps its prediction far more often than R3D-50 did** (about 95% at every m, against 85% at m = 8 for R3D-50). Losing a copy of other contents hurts R3D-18 less.
- **Showing every content twice is almost harmless.** The reference changes the prediction in only 4.3% of videos.

The per-pair gate (see `R3D_E1_summary.md`, section 4) keeps 96%, 96% and 94% of pairs at m = 4, 6 and 8.

## 5. Results

### 5.1 Credit per copy and credit in total

- **per-copy ratio:** the credit of one copy, divided by its credit in the reference;
- **total ratio:** the credit of all copies added up, divided by the total in the reference.

Reference points: if the credit is split evenly between the copies, each copy gets 2/m (0.50, 0.33, 0.25) and the total stays at 1. If each copy keeps its full credit, each gets 1 and the total grows to m/2 (2, 3, 4).

Exact copies, medians over targets:

| method | per-copy ratio, m = 4 / 6 / 8 | total ratio, m = 4 / 6 / 8 |
|---|---|---|
| (credit split evenly) | 0.50 / 0.33 / 0.25 | 1.00 / 1.00 / 1.00 |
| **(how much the model relies on t, I_t ratio)** | – | **1.06 / 1.16 / 1.30** |
| Shapley | 0.72 / 0.60 / 0.52 | 1.44 / 1.79 / 2.07 |
| Integrated Gradients | 0.62 / 0.51 / 0.45 | 1.25 / 1.53 / 1.80 |
| Leave-one-out | 0.07 / 0.09 / −0.02 | 0.15 / 0.26 / −0.08 |
| Occlusion | 0.45 / 0.35 / 0.32 | 0.91 / 1.04 / 1.28 |
| Grad-CAM | 0.87 / 0.82 / 0.71 | 1.74 / 2.47 / 2.84 |

The I_t row uses only the attributed pairs, so it differs a little from section 4.

### 5.2 What each method does

- **Leave-one-out gives each copy almost no credit** (0.07 at m = 4, about 0 at m = 8). Removing one copy leaves the others in the clip, so the model's output hardly changes. The same as R3D-50.
- **Shapley lowers each copy's credit, but raises the total.** Each copy drops to 0.52 at m = 8, and the copies together get 2.07 times the reference credit. The model relies on the target only 1.30 times as much. The same pattern as R3D-50 (0.62 per copy, 2.47 in total).
- **Integrated Gradients lowers each copy's credit**, to 0.45 at m = 8. Unlike R3D-50, where the total stayed at 1, the total grows to 1.80. That's still closer to the model (1.30) than Shapley's total.
- **Occlusion lowers each copy's credit** to 0.32, and its total (0.91 to 1.28) stays close to the model's. On R3D-50, occlusion showed no clear effect.
- **Grad-CAM raises the total** to 2.84, as on R3D-50.

### 5.3 Is the change caused by duplicating the target?

The dilution slope β describes how fast the credit per copy falls as m grows: β = −1 means the credit is split evenly, β = 0 means each copy keeps its full credit. A clearly negative β for the target, with β near 0 for the control, means the change is caused by duplication.

| method | β, target (exact) [95% range] | β, target (noisy) | β, control (exact / noisy) | caused by duplication? |
|---|---|---|---|---|
| Shapley | −0.47 [−0.50, −0.44] | −0.48 | +0.03 / +0.03 | yes |
| Integrated Gradients | −0.58 [−0.66, −0.50] | −0.58 | +0.07 / +0.06 | yes |
| Leave-one-out | −1.55 [−1.69, −1.38] | −1.50 | −0.09 / −0.08 | yes (credit collapses) |
| Occlusion | −0.54 [−0.64, −0.16] | −0.57 | −0.08 / −0.05 | yes |
| Grad-CAM | −0.23 [−0.27, −0.17] | −0.24 | +0.06 / +0.03 | yes |

Notes:
- β only uses targets whose reference credit is positive. Number of targets per row: Shapley 382 (both seeds counted), Grad-CAM 195, Integrated Gradients 180, leave-one-out 160, occlusion 121.
- For leave-one-out, β below −1 just means the credit went to almost 0.
- On R3D-18, the controls are flat for every method, so every method's target change is caused by duplication. On R3D-50, occlusion and Grad-CAM changed in the control too.

### 5.4 Does duplication change the ranking of frames?

Extra evidence only, as for R3D-50.
- **Inversion:** a frame ranked clearly below the target in the reference is ranked above it after duplication, while the model still ranks the target higher.
- **Top-1 loss:** the target was the method's top frame in the reference, but isn't any more.

Exact copies:

| method | inversions, target (m = 4 / 6 / 8) | inversions, control | top-1 loss, target | top-1 loss from random noise alone |
|---|---|---|---|---|
| Shapley | 6% / 15% / 23% | 3% / 9% / 9% | 0.48 / 0.62 / 0.75 | 0.38 |
| Integrated Gradients | 20% / 19% / 22% | 16% / 17% / 17% | 0.66 / 0.66 / 0.67 | 0 |
| Leave-one-out | 25% / 27% / 32% | 20% / 21% / 17% | 0.79 / 0.92 / 0.88 | 0 |
| Occlusion | 21% / 19% / 16% | 21% / 24% / 23% | 0.64 / 0.59 / 0.63 | 0 |
| Grad-CAM | 21% / 26% / 33% | 11% / 19% / 25% | 0.37 / 0.31 / 0.37 | 0 |

- **Shapley shows the clearest ranking effect.** Inversions rise from 6% to 23% for the target, against 3–9% for the control. Top-1 loss rises from 0.48 to 0.75, well above the 0.38 that random noise alone gives. This is clearer than on R3D-50 (16% inversions and 0.71 top-1 loss at m = 8).
- **Leave-one-out and Grad-CAM** also have more inversions for the target than for the control.

## 6. Conclusions

1. **R3D-18 repeats the R3D-50 results.** The model keeps its prediction (about 95% of pairs pass the gate) and its reliance on the target stays at 1.06–1.30 times the reference. Yet every method lowers the credit per copy, and the controls stay flat.

2. **The methods fail in the same ways as on R3D-50.**
   - Leave-one-out: almost no credit per copy.
   - Shapley: less credit per copy, but too much in total (2.07 against the model's 1.30).
   - Grad-CAM: too much in total (2.84).
   - Integrated Gradients and occlusion: less credit per copy, with totals closer to the model's.

3. **Duplication changes which frame looks most important.** For Shapley, the duplicated target loses its top rank in 75% of cases at m = 8, against 38% from random noise alone.

4. **Noisy copies give the same results as exact copies** (β within 0.05).

## 7. Limitations

- **Selection.** Only videos the model gets right, with at least one important content: 69% of the videos screened (345 of 500). 120 of them were attributed.
- **The reference isn't the original clip.** It shows 8 frames, each twice. It gives the same prediction as the original for 95.7% of videos.
- **Methods that delete frames, including Play Fair, can't be judged on R3D-18**, for the same reason as R3D-50: its output depends on clip length.
- **Two machines.** The clips and gate checks were computed on a laptop, and the attribution in fp16 on Colab. The reference scores are recomputed in the attribution run, so every ratio compares scores from the same run.

## 8. Files and how to reproduce

Files in `results/e1/`:

| file | what it contains |
|---|---|
| `r3d_18_conditions.jsonl`, `r3d_18_skipped.jsonl` | the clips built for each video (m = 4, 6, 8), and the videos left out |
| `r3d_18_gate.csv` | the gate check (section 4) |
| `r3d_18_attributions.jsonl` | the per-frame scores for every clip, method and seed |
| `r3d_18_rows.csv`, `r3d_18_slopes.csv`, `r3d_18_summary.csv`, `r3d_18_beta.csv` | the measures, and β with 95% ranges |
| `r3d_18_build.log`, `r3d_18_metrics.log` | run logs |
| `../frame_count_tv_*.csv`, `../length_fixed_content_tv_*.csv` | accuracy against clip length, and at fixed content |

Commands:
```
python motivation/e1_build_conditions.py --model r3d_18 --limit 500
python motivation/e1_metrics.py gate --model r3d_18
python motivation/e1_attribute.py --model r3d_18 --limit 120 --fp16
python motivation/e1_metrics.py attr --model r3d_18
```
