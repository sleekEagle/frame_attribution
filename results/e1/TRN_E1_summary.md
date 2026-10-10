# E1 summary: TRN (multi-scale TRN, BN-Inception, Something-Something v2)

**Status:** run complete for `trn_official` with 5 attribution methods (Shapley-freeze, leave-one-out-freeze, occlusion, Integrated Gradients, Grad-CAM), on all kept videos. **Play Fair's TRN (`trn`) could not be run:** its checkpoints (`trn.pth`, `trn_8_frames.pth`) have been deleted from the authors' Dropbox, and the local copies are the Dropbox "File Deleted" HTML pages. **Play Fair itself was not run on `trn_official`** (see [Pending](#pending)).

**Question.** When a frame the model relies on appears m times in a clip, do frame-attribution methods still assign that content the attribution that the model's own behaviour supports?

## Setup

| item | value |
|---|---|
| model | `trn_official`: the original TRN-pytorch multi-scale checkpoint (`TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar`), BN-Inception per-frame features, relation heads over 2–8 frames, 174 SSv2 classes, **fixed 8-frame input** |
| data | `ssv2_sampled`: 1,044 SSv2 videos (6 per class × 174 classes, `dataloaders/ssv2_paths.txt`), all screened |
| design | **reallocation**: the clip has a fixed 8 slots. K = 4 contents (every other of the 8 sampled frames); the reference presents each content in two slots (m_ref = 2). The target t is given m slots, freed by m − 2 other contents (*losers*) being reduced to one slot. The **control** gives the same freed slots to the content with the smallest model-based importance (the *recipient*), and t keeps two slots. Insertion isn't possible because the relation heads require exactly 8 inputs. |
| m | 2 (reference) and **4**, the only level the 8-slot design permits (m ≤ K). At m = 4 the target clip contains t × 4, the recipient × 2, and the two remaining contents × 1 each. |
| copy types | `exact` (identical) and `noise` (Gaussian, σ = 2/255) |
| frame removal | `late` freeze: a removed slot is filled with the nearest remaining frame before it and, in a second input, the nearest after it; the model's probabilities on the two inputs are averaged. Every input therefore has 8 frames. |
| determinism | The relation module's random subsampling of frame subsets is fixed by the E1 adapter (numpy seed 0 around each call), so repeated evaluations of the same input give identical outputs. |
| targets | Selected by the **model-based importance** I(c): the decrease in the predicted-class probability when every copy of content c is removed. One *high* target (top 2) and, where available, one *mid* target (rank 3–4), both with I(c) > 0.01. No attribution method is involved. |
| gate | A target–control pair is analysed only if **both** clips produce the reference prediction. |
| methods | Shapley (freeze), computed **exactly** over the 8 slots (single seed: a second seed would give identical values); leave-one-out (freeze); whole-frame occlusion; Integrated Gradients (32 steps, zero baseline); Grad-CAM on the per-frame features |

## Sample

| stage | count |
|---|---|
| videos screened | 1,044 |
| excluded: misclassified on the reference clip | 778 |
| excluded: no content with I(c) > 0.01 | 21 |
| **kept** | **245 videos, 375 targets** (245 high, 130 mid) |
| target–control pairs passing the gate | **202 / 375 (53.9%)** exact; 200 / 375 near-duplicate |
| videos contributing at least one passing pair | 166 (exact), 165 (near-duplicate) |
| passing pairs by tier (exact) | high 158 / 245 (64%), mid 44 / 130 (34%) |

The model classifies about 30% of these videos correctly: 30.0% on a 60-video check, 33.6% on the full SSv2 folder in `eval_accuracy.py`. This is why most screened videos are excluded.

## Gate: how much does reallocation change the model's output?

All 375 targets, exact copies. Near-duplicates agree to within 0.5 pp.

| condition | agreement (%) | mean ΔP(pred) | median I_t ratio |
|---|---|---|---|
| original 8 distinct frames vs reference (4 contents × 2) | 87.8 | −0.047 | – |
| target, m = 4 | **72.3** | −0.142 | 1.00 |
| control, m = 4 | **60.0** | −0.241 | 0.58 |

The I_t ratio is the model-based importance of t in the condition divided by its importance in the reference.
- **The reference representation itself changes the prediction in 12.2% of videos** (6% for R3D). TRN combines frame features in temporal order through its relation heads, so presenting 4 contents in consecutive pairs instead of 8 distinct frames affects it more.
- **At m = 4 the perturbation is large.** Half of the non-target content is reduced to a single slot. In the control, the least important content occupies 4 of the 8 slots, which explains why the control is perturbed more than the target.
- **The model's dependence on t is unchanged in the target condition** (I_t ratio 1.00 over all targets). In the analysed pairs it is **1.32** in the target condition and 0.98 in the control.
- **Because only one duplication level exists, the gate can't be relaxed by restricting m.** The per-pair rule is applied as for R3D, retaining 54% of pairs.

## Results

### Quantities

For each method, let the target's attribution be computed for each of its copies; each quantity is relative to the same method's value for t in the reference clip.
- **Per-copy ratio:** the mean attribution of one copy of t. Under exact division of a fixed total among the copies it equals 2/m = **0.50**.
- **Total ratio:** the summed attribution of all copies of t. It equals 1 if the total is preserved, and m/2 = 2 if each copy keeps its reference value.
- **β = log(per-copy ratio) / log(m/2).** With a single level m = 4, β is a transformed per-copy ratio, not a slope fitted across several m: β = −1 corresponds to 0.50, and β = 0 to 1.0.

### 1. Per-copy and total attribution

Exact copies, medians over the 202 analysed pairs. Near-duplicates agree to within 0.03 (per-copy) and 0.06 (β).

| method | per-copy ratio | max over copies (ratio) | total ratio | target β [95% CI] | control per-copy ratio | control β |
|---|---|---|---|---|---|---|
| **Shapley (freeze)** | **0.50** | 0.56 | **0.99** | −0.94 [−1.11, −0.81] | 0.98 | −0.03 |
| **Leave-one-out (freeze)** | **0.02** | 0.08 | 0.04 | −2.60 [−3.19, −2.25] | 1.03 | +0.25 |
| **Occlusion** | **0.55** | 0.97 | 1.11 | −0.75 [−0.94, −0.52] | 1.09 | +0.17 |
| Integrated Gradients | 0.84 | 1.05 | 1.68 | −0.21 [−0.28, −0.16] | 0.97 | −0.02 |
| Grad-CAM | 0.96 | 0.96 | **1.92** | −0.05 [−0.07, −0.02] | 1.00 | −0.00 |

n for β (targets with a positive reference attribution): Shapley 200, IG 192, Grad-CAM 182, LOO 180, occlusion 170. For leave-one-out, β values below −1 reflect per-copy attributions close to zero rather than a meaningful rate; the per-copy ratio is the appropriate measure.

### 2. Ranking metrics

Exact copies.
- **Validated inversion:** the share of contents that the method ranked clearly below t in the reference and ranks above t in the condition, counted only where the model's own importance still ranks t higher.
- **Top-1 loss:** among cases where t was the method's highest-ranked content in the reference, the share where it is no longer highest-ranked. All methods here are deterministic, so the seed-noise floor is 0.

| method | validated inversion, target | validated inversion, control | top-1 loss, target (n) | top-1 loss, control |
|---|---|---|---|---|
| Shapley (freeze) | **36.4%** | 7.8% | **0.61** (105) | 0.31 |
| Leave-one-out (freeze) | **39.0%** | 17.4% | **0.83** (102) | 0.57 |
| Occlusion | 16.7% | 14.6% | 0.43 (69) | 0.19 |
| Integrated Gradients | 9.5% | 10.4% | 0.19 (69) | 0.16 |
| Grad-CAM | 14.6% | 5.7% | 0.25 (68) | 0.16 |

The control's top-1 losses are not zero because the control clip also differs from the reference: the losers are reduced and the recipient is duplicated.

## Comparison with R3D

Per-copy ratio at m = 4 (exact copies). The value under exact division is 0.50.

| method | R3D | TRN | consistent across models? |
|---|---|---|---|
| Shapley (freeze) | 0.76 | 0.50 | no: partial division (R3D) vs exact division (TRN) |
| Integrated Gradients | 0.54 | 0.84 | no: near-exact division (R3D) vs slight reduction (TRN) |
| Leave-one-out (freeze) | 0.05 | 0.02 | yes: approximately zero |
| Occlusion | 0.69, not duplication-specific | 0.55, duplication-specific | no |
| Grad-CAM | 0.91 | 0.96 | yes: little change per copy, so the total increases |

## Conclusions

**1. Duplication changes the attribution of a content although the model's dependence on that content is unchanged.**

In the analysed pairs, the model-based importance of t in the target condition is 1.32 times its reference value, so the model depends on t at least as much as before. For Shapley, leave-one-out and occlusion, the attribution of each copy of t falls well below its reference value. In the matched control conditions, where an unimportant content is duplicated and t keeps two slots, the attribution of t is essentially unchanged (control per-copy ratios 0.97–1.09). The reductions are therefore caused by the duplication of t.

**2. The methods differ in how duplication changes the attribution.**

- **Shapley (freeze) divides a fixed total equally among the copies.** The per-copy ratio equals 2/m (0.50), and the total is preserved (0.99). Each copy is reported as half as important as in the reference, while the model's dependence on t has slightly increased. Duplication also displaces t in Shapley's ranking: validated inversions rise to 36% (8% in the control), and t loses the top rank in 61% of cases.
- **Leave-one-out (freeze) assigns approximately zero attribution to each copy** (2% of the reference value). When one copy is removed, the remaining copies are still present, so the output is almost unchanged. A content the model depends on is reported as unimportant. It has the highest inversion rate (39%) and top-1 loss (0.83).
- **Occlusion approximately divides the total among the copies** (per-copy 0.55, total 1.11), with no comparable change in the control.
- **Integrated Gradients reduces the per-copy attribution only slightly** (0.84), so the total attribution of the content increases to 1.68 times its reference value. That exceeds the model's dependence (1.32).
- **Grad-CAM leaves the per-copy attribution essentially unchanged** (0.96), so the total approximately doubles (1.92).

**3. None of the methods is consistent with the model on both measures.**

Either the per-copy attribution falls well below its reference value (Shapley, leave-one-out, occlusion), or the total attribution exceeds the model's dependence (Integrated Gradients, Grad-CAM). The attribution assigned to a content therefore depends on how many times it occurs in the input, not only on how much the model depends on it.

**4. The direction of the distortion depends on the model, not only on the method.**

Across R3D and TRN, leave-one-out consistently assigns approximately zero attribution to copies, and Grad-CAM consistently inflates the total. For Shapley, Integrated Gradients and occlusion, however, the magnitude of the per-copy reduction differs substantially between the two models (table above). A method's behaviour under duplication can therefore not be characterised independently of the model it explains.

**5. The effects don't depend on the copies being pixel-identical.**

Near-duplicates (σ = 2/255) give the same per-copy ratios to within 0.03.

## Limitations

- **One duplication level (m = 4).** The 8-slot design admits no other level, so the results are single ratios, not slopes across several m.
- **Strong selection.**
  - Only 245 of 1,044 videos are correctly classified and contain an important content.
  - Of their 375 target–control pairs, only 54% keep the reference prediction in both clips, and mid-tier targets pass less often (34%) than high-tier targets (64%).
  - The analysed pairs are therefore those whose prediction survives a large perturbation.
- **The reference is not the original clip.** It contains 4 contents, each in two consecutive slots; it changes TRN's prediction in 12% of videos.
- **The control is strongly perturbed** (60% agreement): in the control clip the least important content occupies half the slots.
- **Only `trn_official`.** Play Fair's TRN checkpoint is no longer available.
- **No Play Fair or drop-based methods.** The E1 adapter requires exactly 8 inputs.

## Pending

1. **Play Fair on `trn_official`.** Play Fair evaluates a subset of k frames with the relation head for k frames; `trn_official` has heads for 2–8 frames. This requires a new subset-evaluation path in the E1 adapter. It would also require a check of the model's accuracy as a function of k, analogous to the input-length check for R3D, because the attributions would then depend on the behaviour of the smaller-scale heads.
2. **`trn` (Play Fair's TRN),** if copies of `trn.pth` and `trn_8_frames.pth` can be obtained (e.g. from the Play Fair authors). Place them in `play-fair/checkpoints/backbones/` and `play-fair/checkpoints/features/`; the commands below then apply with `--model trn`.

## Files (`results/e1/`)

| file | contents |
|---|---|
| `trn_official_conditions.jsonl`, `trn_official_skipped.jsonl` | conditions and model statistics for the 245 kept videos; excluded videos with reasons |
| `trn_official_gate.csv` | gate report |
| `trn_official_attributions.jsonl` | per-slot attributions for every analysed condition and method |
| `trn_official_rows.csv`, `trn_official_slopes.csv`, `trn_official_summary.csv`, `trn_official_beta.csv` | metrics: per row, per target, per method × condition, and β with bootstrap CIs |
| `trn_official_build.log`, `trn_official_attribute.log` | run logs |

## Reproduce

From the repo root, in `tc_env`. To start fresh, move existing `results/e1/trn_official_*` files away first, because every step resumes from existing output.
```
python motivation/e1_build_conditions.py --model trn_official      # all 1,044 videos (~6 min)
python motivation/e1_metrics.py gate --model trn_official
python motivation/e1_attribute.py --model trn_official --seeds 0   # ~2 h
python motivation/e1_metrics.py attr --model trn_official
```
Requires `weights_only=False` in `models/trn_official.py` (PyTorch ≥ 2.6) and the TRN-pytorch checkpoint in `play-fair/checkpoints/backbones/`.
