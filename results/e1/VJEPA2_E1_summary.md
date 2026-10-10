# E1 summary: V-JEPA2 on Something-Something v2

## In short

We tested what attribution methods do when an important frame appears several times in a clip, on V-JEPA2, a large video transformer.

V-JEPA2's accuracy doesn't depend on clip length when the content is the same. So on this model we can add copies (the clip gets longer) and use methods that delete frames, without the length problem that R3D has.

The model barely changes when a frame is duplicated. It keeps its prediction in 95–97% of cases, and it relies on the duplicated frame about as much as before (0.87–0.96 times).

The attribution methods do change:
- **Shapley (with deleted frames)** gives each copy less credit, but the copies about 1.5 times the reference credit in total.
- **Occlusion and Integrated Gradients** give each copy almost no credit.
- **Leave-one-out** keeps 20–30% of the credit per copy.
- **Grad-CAM** keeps most of the credit per copy, so the total grows to 3–7 times.

So no method gives the duplicated frame credit that matches how the model uses it.

## 1. The question

When a frame the model relies on appears m times in a clip, does an attribution method still give it the right amount of credit?

"The right amount" is judged against the model itself. We measure how much the model relies on the frame before and after duplication (the I_t ratio). If the model relies on it as much as before, the frame's credit shouldn't change just because it's repeated.

## 2. Does clip length matter for V-JEPA2?

Our design adds copies, which makes the clip longer. Methods that delete frames show the model clips of 1 to 24 frames. Both are only fair if clip length alone doesn't change the model's output. We checked this on 500 SSv2 videos.

**Accuracy against clip length.** Here the frames are sampled evenly, so content and length change together:

| frames | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 15 | 16 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy (%) | 12.2 | 26.0 | 47.2 | 57.0 | 64.6 | 66.8 | 67.6 | 69.0 | 67.2 | 68.2 | 68.4 | 68.2 | 67.8 |

Accuracy rises smoothly and then levels off. It's flat from 10 to 24 frames, with no sudden jumps. R3D, in contrast, jumps by +28.4 points between 8 and 9 frames, and drops by 10.2 points between 16 and 17.

**Accuracy with the same content.** We took k different frames and showed them in two ways: as a short clip of k frames, and with each frame repeated to fill 16 slots.

| k different frames | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy as a short clip (%) | 12.2 | 26.0 | 47.2 | 57.0 | 64.6 | 66.8 | 67.6 | 69.0 | 68.2 |
| accuracy repeated to 16 (%) | 14.4 | 24.4 | 50.4 | 60.8 | 62.4 | 66.2 | 68.6 | 67.8 | 68.2 |
| difference (points) | +2.2 | −1.6 | +3.2 | +3.8 | −2.2 | −0.6 | +1.0 | −1.2 | 0 |
| same prediction both ways (%) | 60.8 | 31.2 | 53.6 | 67.2 | 71.8 | 81.8 | 85.8 | 88.2 | 100 |

What this shows:
- **Length alone doesn't change accuracy.** The two ways differ by at most 3.8 points, with no consistent direction. The probability of the true class differs by at most 0.033. For R3D, the repeated clips were 41–52 points more accurate.
- **What matters is how many different frames the model sees.** One frame repeated 16 times gives only 14.4%. SSv2 classes depend on motion, unlike UCF101, where R3D gets 63.8% from one repeated frame.
- **With few frames, the predicted class often differs between the two ways** (31–67% agreement at k = 2–6), but accuracy doesn't. With so little information, the model's probabilities are low (0.2–0.5 for the true class), so the top class flips between close alternatives.

**Why this matters.** When a method deletes frames, adding a copy back makes the clip one frame longer without adding new content. Since length alone doesn't change V-JEPA2's output, the copy's credit reflects its content, not the length change. So **methods that delete frames, including Play Fair, can be judged fairly on V-JEPA2.** The flipping predictions at small clips add noise, but no bias towards copies.

## 3. How the test works

**Model.** `facebook/vjepa2-vitl-fpc16-256-ssv2`: a ViT-L that takes 16 frames at 256 px. It groups frames in pairs: each pair of frames becomes one set of tokens.

**Data.** `ssv2_sampled`: 1,044 SSv2 videos, 6 per class. We screened 72, with classes sampled evenly. The run was done in Google Colab.

**Clips.** We use the *insertion* design:
- **Reference clip:** the original 16 frames, each once (m = 1).
- **Target clip:** m − 1 extra copies of the target are inserted next to it. They're inserted as whole frame pairs, so the pairs of all other frames stay the same. The clip gets longer: 18, 20 or 24 frames.
- **Control clip:** the same number of extra frames, but as copies of frame pairs of other contents. The target appears once. The control has the same length as the target clip.

We test m = 3, 5 and 9. These are odd numbers so that the m − 1 inserted copies fill whole frame pairs. Each copy is either exact, or has a little random noise added (σ = 2/255).

**Removing frames.** Removed frames are deleted. The model sees the remaining frames, in order.

**Choosing the target.** We measure how much the model needs each frame: we remove every copy of it and see how much the predicted class's probability drops. This is the frame's *importance* I. One target is picked from the top 2 frames, and a second from ranks 3–8 when available. Both need I > 0.01. No attribution method is used to choose them.

**The per-pair gate.** A target clip and its control are used only if both give the reference prediction.

**Methods:**
- Shapley-drop: 32 random orderings (each with its reverse), 2 seeds;
- leave-one-out (drop);
- occlusion (one frame replaced by a blank frame);
- Integrated Gradients (32 steps, blank baseline, full precision);
- Grad-CAM (last encoder block).

Shapley and leave-one-out ran the model in half precision (fp16).

## 4. Which videos were used

| step | number |
|---|---|
| videos screened | 72 |
| left out: the model got them wrong | 26 |
| left out: no single frame mattered enough (I ≤ 0.01) | 12 |
| **kept** | **34 videos, 57 targets** (34 top, 23 middle) |
| target–control pairs that passed the gate, exact copies | 53 / 52 / 51 of 57 (93% / 91% / 90%) at m = 3 / 5 / 9 |
| the same, noisy copies | 54 / 51 / 52 of 57 |
| videos with at least one passing pair | 33 |

## 5. Does duplication change the model?

All 57 targets:

| clip | m = 3 | m = 5 | m = 9 |
|---|---|---|---|
| target clip: same prediction (%), exact / noisy | 94.7 / 96.5 | 96.5 / 96.5 | 94.7 / 94.7 |
| target clip: change in the predicted class's probability | −0.023 | −0.025 | −0.049 |
| target clip: how much the model relies on t (I_t ratio) | 0.95 | 0.94 | 0.85 |
| control clip: same prediction (%), exact / noisy | 96.5 / 96.5 | 91.2 / 89.5 | 89.5 / 91.2 |
| control clip: change in the predicted class's probability | −0.032 | −0.057 | −0.086 |

Adding up to 8 copies of the target changes the prediction in at most 5% of cases, and lowers the predicted class's probability by at most 0.05 on average. In the pairs used for attribution, the model relies on the target 0.96, 0.95 and 0.87 times as much as in the reference. **So the model relies on the duplicated frame about as much as before**, a little less at m = 9.

## 6. Results

### 6.1 Credit per copy and credit in total

In the insertion design, the reference has **one** copy of the target. So the reference points differ from R3D and TRN:
- **per-copy ratio:** the credit of one copy, divided by its credit in the reference. If the credit is split evenly, each copy gets 1/m: **0.33, 0.20, 0.11** at m = 3, 5, 9.
- **total ratio:** the credit of all copies added up. It stays at 1 if the credit is split evenly, and grows to m (3, 5, 9) if each copy keeps its full credit.
- **best copy:** the credit of the copy with the highest credit.

Exact copies, medians over the pairs used (Shapley: both seeds):

| method | per-copy ratio, m = 3 / 5 / 9 | best copy, m = 3 / 5 / 9 | total ratio, m = 3 / 5 / 9 |
|---|---|---|---|
| (credit split evenly) | 0.33 / 0.20 / 0.11 | | 1 / 1 / 1 |
| **(how much the model relies on t)** | | | **0.96 / 0.95 / 0.87** |
| Shapley-drop | 0.50 / 0.30 / 0.16 | 0.77 / 0.70 / 0.69 | **1.51 / 1.50 / 1.45** |
| Leave-one-out | 0.22 / 0.20 / 0.30 | 0.22 / 0.20 / 0.30 | 0.65 / 0.99 / 2.69 |
| Occlusion | **0.09 / 0.04 / 0.00** | 0.51 / 0.38 / 0.47 | 0.28 / 0.20 / 0.01 |
| Integrated Gradients | **0.11 / 0.04 / 0.00** | 0.36 / 0.47 / 0.72 | 0.34 / 0.20 / 0.04 |
| Grad-CAM | 0.97 / 0.90 / 0.74 | 1.15 / 1.06 / 0.93 | **2.92 / 4.48 / 6.67** |

### 6.2 What each method does

- **Shapley-drop splits the credit only partly, so the total is too high.** Each copy gets less than in the reference (0.50, 0.30, 0.16), but more than an even split. Added up, the copies get about **1.5 times** the reference credit at every m. The model relies on the target only 0.87–0.96 times as much. The copies aren't treated alike: the best copy keeps about 70% of the reference credit. A likely reason is the frame pairs: some copies share a pair with another copy, others with a neighbouring frame.
- **Occlusion gives each copy almost no credit** (0.09, 0.04, 0.00). Blanking one copy leaves the others in the clip, so the output barely changes. On this model, occlusion behaves like leave-one-out does on R3D and TRN.
- **Leave-one-out keeps 20–30% per copy, without going to 0.** This differs from R3D and TRN, where it collapses. Deleting one frame changes how all later frames are paired into tokens, so the removal changes more than one copy. In the control, the target's credit also falls (0.77, 0.54, 0.83), which fits this explanation.
- **Integrated Gradients gives each copy almost no credit** (0.11, 0.04, 0.00). But the target's credit **also falls in the control** (to 0.09–0.27), where the target isn't duplicated. So on V-JEPA2, Integrated Gradients reacts strongly to any change of the clip. Only part of its fall is caused by duplication.
- **Grad-CAM barely lowers the credit per copy**, so the total grows almost with m (2.92, 4.48, 6.67). Its small fall at m = 9 also happens in the control (0.74 against 0.72), so it isn't caused by duplication. The same as on R3D and TRN.

### 6.3 Is the change caused by duplicating the target?

The dilution slope β describes how fast the credit per copy falls as m grows: β = −1 means the credit is split evenly, β = 0 means each copy keeps its full credit. A clearly negative β for the target, with β near 0 for the control, means duplication causes the change.

| method | β, target (exact) [95% range] | control per-copy ratio, m = 3 / 5 / 9 | β, control | caused by duplication? |
|---|---|---|---|---|
| Shapley-drop | −0.71 [−0.86, −0.55] | 1.00 / 1.10 / 0.86 | −0.02 | yes |
| Leave-one-out | −0.48 [−0.69, −0.22] | 0.77 / 0.54 / 0.83 | −0.05 | yes |
| Occlusion | −0.75 [−1.21, −0.58] | 1.00 / 1.01 / 1.14 | +0.09 | yes |
| Integrated Gradients | −1.45 [−1.98, −1.24] | 0.12 / 0.27 / 0.09 | −0.46 [−0.88, −0.12] | partly (the control falls too) |
| Grad-CAM | −0.08 [−0.17, −0.02] | 0.94 / 0.88 / 0.72 | −0.11 | no (target and control change alike) |

Notes:
- Noisy copies give the same pattern. β differs by at most 0.27, most for Integrated Gradients.
- β only uses targets whose reference credit is positive: Shapley 98 (both seeds counted), Grad-CAM 54, leave-one-out 54, occlusion 45, Integrated Gradients 34.
- For occlusion, Integrated Gradients and leave-one-out, β below −1 or per-copy ratios near 0 just mean the credit went to almost 0. The per-copy and total ratios describe them better.

### 6.4 Does duplication change the ranking of frames?

Extra evidence only, because there are few cases.
- **Inversion:** a frame ranked clearly below the target in the reference is ranked above it after duplication, while the model still ranks the target higher.
- **Top-1 loss:** the target was the method's top frame in the reference, but isn't any more.

Exact copies:

| method | inversions, target (m = 3 / 5 / 9) | inversions, control | top-1 loss, target (number of cases) |
|---|---|---|---|
| Shapley-drop | 12.1% / 16.6% / 19.0% | 12.6% / 19.0% / 18.3% | (3 cases) |
| Leave-one-out | 22.3% / 28.7% / 20.6% | 3.6% / 8.0% / 13.7% | 0.91 / 0.95 / 0.95 (22 / 21 / 21) |
| Occlusion | 15.8% / 17.7% / 14.9% | 17.0% / 16.7% / 21.9% | 0.88 / 1.00 / 0.75 (8) |
| Integrated Gradients | 27.8% / 20.4% / 12.2% | 42.3% / 47.9% / 67.3% | (4–5 cases) |
| Grad-CAM | 8.1% / 8.4% / 5.7% | 4.5% / 13.4% / 20.7% | (7 cases) |

Top-1 loss can only be measured where the target was the method's top frame in the reference, which is rare here (3–23 cases). Only leave-one-out has more inversions for the target than for the control.

## 7. Conclusions

1. **Duplication changes a frame's credit, although the model relies on it about as much as before.** V-JEPA2 keeps its prediction for 95–97% of targets, and relies on the target 0.87–0.96 times as much. Yet every method except Grad-CAM changes the copies' credit a lot. For Shapley-drop, occlusion and leave-one-out, the control clips show no such change.

2. **Shapley with deleted frames, the quantity Play Fair computes, splits the credit only partly and gives too much in total.** Each copy gets less than in the reference, but more than an even split. The copies together get about 1.5 times the reference credit, while the model doesn't rely on the target more. V-JEPA2's output doesn't depend on clip length, so unlike on R3D, this can't be explained by a length effect.

3. **No method matches the model on both measures.**
   - Too little credit per copy: occlusion and Integrated Gradients (about 0), leave-one-out (0.2–0.3), Shapley-drop (above an even split, but well below 1).
   - Too much credit in total: Shapley-drop (about 1.5) and Grad-CAM (3–7).

4. **How a method fails depends on the model too.** Grad-CAM inflates the total on R3D, TRN and V-JEPA2. Leave-one-out collapses on R3D and TRN, but keeps 20–30% per copy on V-JEPA2. Occlusion has no duplication effect on R3D-50, but collapses on V-JEPA2.

5. **Noisy copies give the same results as exact copies.**

## 8. Limitations

- **Small sample.** 34 videos and 57 targets, about a third of R3D-50's. The 95% ranges are wider (for example ±0.15 for Shapley-drop's β), and the ranking measures rest on very few cases.
- **Play Fair wasn't run on V-JEPA2.** Shapley-drop computes the same quantity with a different sampler. We showed they agree within random noise on R3D-50 and on MC3-18 (see `MC3_18_E1_summary.md`, section 6). On V-JEPA2, Play Fair would cost much more: about 110 ms per model evaluation, against about 5 ms for MC3-18.
- **Frame pairs.** V-JEPA2 processes frames in pairs. Deleting one frame changes the pairing of all later frames. This affects leave-one-out and the subsets Shapley-drop evaluates. Clips with an odd number of frames are padded by repeating the last frame. The same-content check shows no length effect, but it doesn't separate out the effect of re-pairing.
- **Integrated Gradients** reacts strongly to the control clips, so treat its results on this model with caution.
- **Selection.** Only videos the model gets right, with an important frame, are used: 34 of 72.

## 9. Still to do (optional)

1. **More videos**, if there's compute. The build and attribution steps resume, so `--limit` can simply be raised.
2. **Play Fair on V-JEPA2.** Optional now that MC3-18 confirms Play Fair = Shapley-drop on a model where deletion is valid. It would need `--pf-max-samples 64` and `--fp16` to be affordable.
3. **Shapley and leave-one-out with freeze** (`shapley_freeze`, `loo_freeze`), to compare deleting frames with filling them in, on a model without a length effect.

The other models are covered in their own summaries and in `results/E1_report.md`.

## 10. Files and how to reproduce

Files in `results/e1/`:

| file | what it contains |
|---|---|
| `vjepa2_conditions.jsonl`, `vjepa2_skipped.jsonl` | the clips built for the 34 kept videos, and the videos left out |
| `vjepa2_gate.csv` | the gate check (section 5) |
| `vjepa2_attributions.jsonl` | 4,164 per-frame score sets: 694 clips × (Shapley-drop × 2 seeds + 4 other methods) |
| `vjepa2_rows.csv`, `vjepa2_slopes.csv`, `vjepa2_summary.csv`, `vjepa2_beta.csv` | the measures, and β with 95% ranges |
| `length_fixed_content_vjepa2_*.csv` (this folder) | accuracy for k frames as a short clip vs repeated to 16 |
| `../frame_count_vjepa2_short_*.csv`, `../frame_count_summary.csv` | accuracy against clip length, 1–16 and 16–24 frames |

Commands (Colab; clone the repository, unzip `ssv2_sampled.zip`, set `SSV2_PATH`, `pip install torchcodec`):
```
python -u eval_frame_count.py --models vjepa2 --frames 1 2 4 6 8 10 12 14 15 16 --limit 500 --datasets vjepa2=ssv2_sampled --out {OUT}/frame_count_vjepa2_short
python -u eval_length_fixed_content.py --models vjepa2 --datasets vjepa2=ssv2_sampled --ks 1 2 4 6 8 10 12 14 16 --limit 500 --out {OUT}/length_fixed_content_vjepa2
python -u motivation/e1_build_conditions.py --model vjepa2 --limit 72 --fp16 --out {OUT}
python -u motivation/e1_metrics.py gate --model vjepa2 --out {OUT}
python -u motivation/e1_attribute.py --model vjepa2 --fp16 --methods shapley_drop loo_drop occlusion ig gradcam --n-perm 32 --seeds 0 1 --batch-size 64 --ig-batch 2 --out {OUT}
python -u motivation/e1_metrics.py attr --model vjepa2 --out {OUT}
```
