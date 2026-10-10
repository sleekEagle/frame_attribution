# E1 summary: V-JEPA2 (ViT-L, fpc16, 256 px, Something-Something v2)

**Status:** E1 complete on 72 screened videos (34 kept, 57 targets) with Shapley-drop, leave-one-out-drop, occlusion, Integrated Gradients and Grad-CAM. **Official Play Fair hasn't been run on V-JEPA2 yet** (see [Pending](#pending)); Shapley-drop computes the same quantity (Element Shapley Values) with a different sampling scheme. Run in Google Colab.

**Question.** When a frame the model relies on appears m times in a clip, do frame-attribution methods still assign that content the attribution that the model's own behaviour supports?

## Model behaviour as a function of input length

V-JEPA2 is evaluated with the **insertion** design and with frame removal by **deletion**, so attribution evaluates the model on inputs of 1–24 frames. Three measurements on 500 `ssv2_sampled` videos establish that this is valid.

**1. Accuracy against input length, frames resampled uniformly** (`eval_frame_count.py`; content and length change together):

| frames | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 15 | 16 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy (%) | 12.2 | 26.0 | 47.2 | 57.0 | 64.6 | 66.8 | 67.6 | 69.0 | 67.2 | 68.2 | 68.4 | 68.2 | 67.8 |

Accuracy rises smoothly and saturates: the gain per added frame decreases from 13.8 pp (1 → 2) to about 1 pp (8 → 10). It's flat from 10 to 24 frames, and there's no step change at any length. By contrast, R3D shows steps of +28.4 pp (8 → 9) and −10.2 pp (16 → 17).

**2. Accuracy at fixed content** (`eval_length_fixed_content.py`). The same k distinct frames are evaluated as a k-frame input (*native*) and with each frame repeated in consecutive slots to form a 16-frame input (*repeated*):

| k distinct frames | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy, k-frame input (%) | 12.2 | 26.0 | 47.2 | 57.0 | 64.6 | 66.8 | 67.6 | 69.0 | 68.2 |
| accuracy, repeated to 16 (%) | 14.4 | 24.4 | 50.4 | 60.8 | 62.4 | 66.2 | 68.6 | 67.8 | 68.2 |
| difference (pp) | +2.2 | −1.6 | +3.2 | +3.8 | −2.2 | −0.6 | +1.0 | −1.2 | 0 |
| same prediction (%) | 60.8 | 31.2 | 53.6 | 67.2 | 71.8 | 81.8 | 85.8 | 88.2 | 100 |

The two presentations differ by at most 3.8 pp, with no consistent sign. Mean P(ground truth) differs by at most 0.033. **At fixed content, input length has no systematic effect on V-JEPA2's accuracy.** For R3D, the 16-frame presentation was 41–52 pp more accurate at k ≤ 8. Accuracy is determined by the number of distinct frames: a single frame repeated 16 times gives 14.4%. SSv2 classes require temporal information, unlike UCF101, where R3D reaches 63.8% from one repeated frame.

At small k the two presentations often produce different predicted classes (31–67% agreement at k = 2–6), without a difference in accuracy or confidence. With few distinct frames the model's class probabilities are low (P(gt) ≈ 0.2–0.5), so the highest-scoring class varies between close alternatives.

**Consequence for E1.** Under deletion, adding a copy of t to a subset S lengthens the input by one frame without adding content. Because length at fixed content has no systematic effect, the marginal contribution v(S ∪ {i}) − v(S) of a copy reflects its content, not the change in length. **Deletion-based Shapley values (Shapley-drop, Play Fair) are therefore not confounded by input length on V-JEPA2,** unlike on R3D. The variability of individual predictions at small subsets adds noise to those terms, but not a bias toward copies.

## Setup

| item | value |
|---|---|
| model | `facebook/vjepa2-vitl-fpc16-256-ssv2`; tubelet = 2 (each pair of frames forms one group of tokens); 3D rotary position encoding |
| data | `ssv2_sampled` (1,044 SSv2 videos, 6 per class); 72 screened, sampled evenly across classes |
| design | **insertion**: the reference is the original 16-frame clip (m_ref = 1). In the target condition, m − 1 copies of t are inserted next to it as whole frame pairs, so the original pairs of all other frames are unchanged; inputs have 18, 20 and 24 frames. The **spread control** has the same length, with the extra frames being duplicated pairs of other contents; t appears once. |
| m | 1 (reference), 3, 5, 9 (odd, so the m − 1 inserted copies fill whole frame pairs) |
| copy types | `exact` and `noise` (Gaussian, σ = 2/255) |
| frame removal | deletion (`drop`): removed frames are taken out of the input, and the model is evaluated on the remaining frames in temporal order |
| targets | selected by the model-based importance I(c) (decrease in predicted-class probability when every copy of c is removed): one high target (top 2) and, where available, one mid target (rank 3–8), both with I(c) > 0.01 |
| gate | a target–control pair is analysed only if both clips produce the reference prediction |
| methods | Shapley-drop (32 random orderings incl. reverses, seeds 0 and 1), leave-one-out-drop, whole-frame occlusion, Integrated Gradients (32 steps, zero baseline, fp32), Grad-CAM (last encoder block); model evaluations for Shapley and LOO in fp16 |

## Sample

| stage | count |
|---|---|
| videos screened | 72 |
| excluded: misclassified on the reference clip | 26 |
| excluded: no content with I(c) > 0.01 | 12 |
| **kept** | **34 videos, 57 targets** (34 high, 23 mid) |
| target–control pairs passing the gate, exact copies, m = 3 / 5 / 9 | 53 / 52 / 51 of 57 (93.0 / 91.2 / 89.5%) |
| same, near-duplicates | 54 / 51 / 52 of 57 |
| videos contributing at least one passing pair | 33 |

## Gate

All 57 targets.

| condition | m = 3 | m = 5 | m = 9 |
|---|---|---|---|
| target, agreement (%), exact / noise | 94.7 / 96.5 | 96.5 / 96.5 | 94.7 / 94.7 |
| target, mean ΔP(pred), exact | −0.023 | −0.025 | −0.049 |
| target, median I_t ratio, exact | 0.95 | 0.94 | 0.85 |
| control, agreement (%), exact / noise | 96.5 / 96.5 | 91.2 / 89.5 | 89.5 / 91.2 |
| control, mean ΔP(pred), exact | −0.032 | −0.057 | −0.086 |

Inserting up to 8 copies of t changes V-JEPA2's prediction in at most 5% of targets and lowers the predicted-class probability by at most 0.05 on average. In the analysed pairs, the model-based importance of t relative to the reference is 0.96, 0.95 and 0.87 at m = 3, 5 and 9. **The model's dependence on t is approximately unchanged by duplication,** decreasing slightly at m = 9.

## Results

### Quantities

The target's attribution is computed for each of its m copies, and each quantity is relative to the same method's attribution of t in the reference clip. **In the insertion design the reference contains one copy of t,** so the reference values differ from R3D's and TRN's:
- **per-copy ratio:** the mean attribution of one copy of t. Under exact division of a fixed total among the copies, it equals **1/m: 0.33, 0.20, 0.11** at m = 3, 5, 9.
- **total ratio:** the summed attribution of all copies of t. It equals 1 if the total is preserved, and m (3, 5, 9) if each copy keeps its reference value.
- **max ratio:** the attribution of the highest-attributed copy.
- **β:** slope of log(per-copy ratio) against log m. β = −1 corresponds to exact division, and β = 0 to no change per copy.

### 1. Per-copy and total attribution

Exact copies, medians over analysed pairs (Shapley: both seeds).

| method | per-copy ratio, m = 3 / 5 / 9 | max ratio, m = 3 / 5 / 9 | total ratio, m = 3 / 5 / 9 | target β [95% CI] | control per-copy ratio, m = 3 / 5 / 9 | control β |
|---|---|---|---|---|---|---|
| exact division of a fixed total | 0.33 / 0.20 / 0.11 | | 1 / 1 / 1 | −1 | | |
| **Shapley-drop** | 0.50 / 0.30 / 0.16 | 0.77 / 0.70 / 0.69 | **1.51 / 1.50 / 1.45** | −0.71 [−0.86, −0.55] | 1.00 / 1.10 / 0.86 | −0.02 |
| **Leave-one-out-drop** | 0.22 / 0.20 / 0.30 | 0.22 / 0.20 / 0.30 | 0.65 / 0.99 / 2.69 | −0.48 [−0.69, −0.22] | 0.77 / 0.54 / 0.83 | −0.05 |
| **Occlusion** | **0.09 / 0.04 / 0.00** | 0.51 / 0.38 / 0.47 | 0.28 / 0.20 / 0.01 | −0.75 [−1.21, −0.58] | 1.00 / 1.01 / 1.14 | +0.09 |
| **Integrated Gradients** | **0.11 / 0.04 / 0.00** | 0.36 / 0.47 / 0.72 | 0.34 / 0.20 / 0.04 | −1.45 [−1.98, −1.24] | 0.12 / 0.27 / 0.09 | −0.46 [−0.88, −0.12] |
| Grad-CAM | 0.97 / 0.90 / 0.74 | 1.15 / 1.06 / 0.93 | **2.92 / 4.48 / 6.67** | −0.08 [−0.17, −0.02] | 0.94 / 0.88 / 0.72 | −0.11 |

Near-duplicates give the same pattern; β differs by at most 0.27, the largest difference being for Integrated Gradients. n for β (targets with a positive reference attribution): Shapley 98 (both seeds), Grad-CAM 54, leave-one-out 54, occlusion 45, Integrated Gradients 34. For occlusion, Integrated Gradients and leave-one-out, β values below −1 or per-copy ratios near zero reflect attributions close to zero rather than a meaningful rate; the per-copy and total ratios are the appropriate measures.

### 2. Ranking metrics

Exact copies.

| method | validated inversion, target, m = 3 / 5 / 9 | validated inversion, control | top-1 loss, target (n) |
|---|---|---|---|
| Shapley-drop | 12.1 / 16.6 / 19.0% | 12.6 / 19.0 / 18.3% | (3 cases) |
| Leave-one-out-drop | 22.3 / 28.7 / 20.6% | 3.6 / 8.0 / 13.7% | 0.91 / 0.95 / 0.95 (22 / 21 / 21) |
| Occlusion | 15.8 / 17.7 / 14.9% | 17.0 / 16.7 / 21.9% | 0.88 / 1.00 / 0.75 (8) |
| Integrated Gradients | 27.8 / 20.4 / 12.2% | 42.3 / 47.9 / 67.3% | (4–5 cases) |
| Grad-CAM | 8.1 / 8.4 / 5.7% | 4.5 / 13.4 / 20.7% | (7 cases) |

Top-1 loss is defined only where t was the method's highest-ranked content in the reference, which applies to few targets here (3–23 per cell). Only leave-one-out shows more validated inversions in the target condition than in the control.

## Interpretation by method

- **Shapley-drop divides the attribution only partially, so the total attribution exceeds the model's dependence.** The per-copy ratio (0.50, 0.30, 0.16) decreases with m but stays above 1/m, and the total ratio is about **1.5 at every m**, while the model's dependence on t is 0.87–0.96 of its reference value. The duplicated content as a whole is therefore assigned about 1.5 times its reference attribution. The control is unchanged (β = −0.02), so the effect is caused by the duplication of t. The copies are attributed unequally: the highest-attributed copy keeps about 70% of the reference value. A plausible cause is V-JEPA2's frame pairs: some copies of t share a token group with another copy, others with a neighbouring frame.
- **Occlusion assigns approximately zero attribution to each copy** (per-copy 0.09, 0.04, 0.00; total 0.28 to 0.01). Replacing one copy with a blank frame leaves the other copies in the input, so the output barely changes. In the insertion design, occlusion therefore behaves like leave-one-out on R3D and TRN. The control is unchanged.
- **Leave-one-out-drop reduces each copy to about 20–30% of its reference value without collapsing to zero,** unlike leave-one-out-freeze on R3D and TRN. Deleting one frame from the input changes how all subsequent frames are grouped into frame pairs (tokens), so its removal changes the input beyond the loss of one copy. The control's attribution of t also decreases (0.77, 0.54, 0.83), which is consistent with this.
- **Integrated Gradients reduces the per-copy attribution to near zero** (0.11, 0.04, 0.00), so the total falls to 0.34–0.04. However, **t's attribution also falls strongly in the control** (per-copy 0.09–0.27), where t isn't duplicated, and inversions in the control are higher than in the target (42–67%). Integrated Gradients' attribution of t on V-JEPA2 is therefore highly sensitive to any modification of the input, and only part of its reduction is specific to duplication (target β −1.45 against control β −0.46; the confidence intervals don't overlap).
- **Grad-CAM leaves the per-copy attribution largely unchanged,** so the total grows almost in proportion to m (2.92, 4.48, 6.67). Its per-copy decline at m = 9 also occurs in the control (0.74 vs 0.72), so it isn't specific to duplication. This matches its behaviour on R3D and TRN.

## Conclusions

**1. Duplication changes the attribution of a content although the model's dependence on it is approximately unchanged.** V-JEPA2 keeps its prediction for 95–97% of targets under duplication, and its model-based dependence on t remains at 0.87–0.96 of the reference value. Yet every method except Grad-CAM changes the attribution of t's copies substantially, and for Shapley-drop, occlusion and leave-one-out the matched controls show no comparable change.

**2. Deletion-based Shapley values (the quantity computed by Play Fair) partially divide the attribution among the copies and over-attribute the duplicated content in total.** Each copy receives less than in the reference, but more than 1/m. The total is about 1.5 times the reference attribution, while the model's dependence on t isn't increased. V-JEPA2's output doesn't depend on input length at fixed content, so this result isn't explained by a length effect, as it partly was on R3D.

**3. None of the methods is consistent with the model on both the per-copy and the total measure.**
- **Per copy too low:** occlusion and Integrated Gradients (≈ 0), leave-one-out (0.2–0.3), Shapley-drop (above 1/m but well below 1).
- **Total too high:** Shapley-drop (≈ 1.5) and Grad-CAM (3–7).

**4. The direction and size of the distortion depend on the model as well as the method.** Grad-CAM inflates the total on all three models. Leave-one-out collapses on R3D and TRN, but on V-JEPA2 its deletion-based version retains 20–30% per copy. Occlusion has no duplication-specific effect on R3D, but collapses on V-JEPA2.

**5. The effects hold for near-duplicates**, with the same pattern for exact and noisy copies.

## Limitations

- **Small sample:** 34 videos and 57 targets, about a third of R3D's. Confidence intervals are correspondingly wider (e.g. ±0.15 for Shapley-drop's β), and the ranking metrics, especially top-1 loss, rest on very few cases.
- **No official Play Fair run.** Shapley-drop estimates the same quantity (Element Shapley Values, with the same characteristic function) using permutation sampling instead of Play Fair's per-size sampler. The equivalence was verified exactly on short clips and empirically on R3D, but not yet on V-JEPA2.
- **Frame-pair structure.** V-JEPA2 processes frames in pairs. Under deletion, removing a single frame changes the pairing of all later frames, which affects leave-one-out-drop and the subsets evaluated by Shapley-drop. Odd-length inputs are padded by repeating the last frame. The fixed-content check shows no systematic length effect, but it doesn't isolate the effect of re-pairing.
- **Integrated Gradients** uses 32 integration steps with a zero baseline; its strong sensitivity to the control modification suggests it should be interpreted with caution on this model.
- **Selection:** only correctly classified videos with an important content are analysed (34 of 72).

## Pending

1. **Official Play Fair on V-JEPA2,** with `--pf-max-samples 64` and fp16 (requires passing `--fp16` through to `attr_playfair`), to report Play Fair directly and to compare it with Shapley-drop on identical conditions.
2. **A larger sample,** if compute allows (the build and attribution resume, so `--limit` can be increased).
3. **Freeze-based Shapley on V-JEPA2** (`shapley_freeze`, `loo_freeze`), to compare removal by deletion and by replacement on a model without a length effect.

## Files

| file | contents |
|---|---|
| `vjepa2_conditions.jsonl`, `vjepa2_skipped.jsonl` | conditions and model statistics (34 kept videos); excluded videos with reasons |
| `vjepa2_gate.csv` | gate report |
| `vjepa2_attributions.jsonl` | 4,164 per-slot attributions: 694 analysed conditions × (Shapley-drop × 2 seeds + 4 deterministic methods) |
| `vjepa2_rows.csv`, `vjepa2_slopes.csv`, `vjepa2_summary.csv`, `vjepa2_beta.csv` | metrics: per row, per target, per method × condition × m, β with bootstrap CIs |
| `length_fixed_content_vjepa2_*.csv` (this folder) | accuracy at fixed content, k-frame vs repeated-to-16 inputs |
| `../frame_count_vjepa2_short_*.csv`, `../frame_count_summary.csv` | accuracy against input length, 1–16 and 16–24 frames |

## Reproduce (Colab)

Setup as in the Colab notes: clone the repository, unzip `ssv2_sampled.zip`, set `SSV2_PATH`, `pip install torchcodec`. Then:
```
python -u eval_frame_count.py --models vjepa2 --frames 1 2 4 6 8 10 12 14 15 16 --limit 500 --datasets vjepa2=ssv2_sampled --out {OUT}/frame_count_vjepa2_short
python -u eval_length_fixed_content.py --models vjepa2 --datasets vjepa2=ssv2_sampled --ks 1 2 4 6 8 10 12 14 16 --limit 500 --out {OUT}/length_fixed_content_vjepa2
python -u motivation/e1_build_conditions.py --model vjepa2 --limit 72 --fp16 --out {OUT}
python -u motivation/e1_metrics.py gate --model vjepa2 --out {OUT}
python -u motivation/e1_attribute.py --model vjepa2 --fp16 --methods shapley_drop loo_drop occlusion ig gradcam --n-perm 32 --seeds 0 1 --batch-size 64 --ig-batch 2 --out {OUT}
python -u motivation/e1_metrics.py attr --model vjepa2 --out {OUT}
```
