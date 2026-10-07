# E1 summary: R3D (ResNet3D-50, UCF101)

**Status:** 5 freeze/gradient methods on 120 videos, plus Shapley-drop and LOO-drop on the same 120 videos, and official Play Fair on the first 20. The Play Fair results are a reduced run: 256 samples per subset size and 1 seed (see [Drop-based methods and Play Fair](#drop-based-methods-and-play-fair)).

**Question.** When a frame the model relies on appears m times in a clip, do frame-attribution methods still give that content the credit the model's own behaviour supports?

## Setup

| item | value |
|---|---|
| model | R3D-50 fine-tuned on UCF101 (`models/r3d/ckpt/save_200.pth`), 16-frame input |
| data | UCF101 split-1 test set (`ucf101_test`), classes sampled evenly |
| design | **reallocation**: the clip stays at 16 slots. 8 contents (every other sampled frame); the reference shows each twice (m_ref = 2). The target gets m slots, freed by m − 2 *losers* dropping to one copy. The **control** gives those same slots to the least important content instead. See `results/design_choice_report.md` for why R3D can't use insertion. |
| m | 2 (reference), 4, 6, 8 |
| copy types | `exact` (identical) and `noise` (Gaussian, σ = 2/255) |
| frame removal | `late` freeze: the removed slot is filled with the average of filling from the frame before and the frame after, so length stays 16 |
| targets | chosen by the **model's** content importance I(c) (P(pred) drop when every copy of c is removed), never by an attribution method: one *high* target (top 2) and, where available, one *mid* target (ranks 3–4), both with I > 0.01 |
| gate | a target/control pair is kept only if **both** clips keep the reference prediction |
| methods | Shapley (freeze, 64 orderings × 2 seeds), leave-one-out (freeze), whole-frame occlusion, Integrated Gradients (32 steps, zero baseline), Grad-CAM (`layer2`) |

## Sample

| stage | count |
|---|---|
| test videos sampled | 500 |
| skipped: misclassified | 102 |
| skipped: no frame with I > 0.01 | 178 |
| kept (conditions + gate) | 220 videos, 343 targets |
| **attributed** (first 120 kept videos) | **120 videos, 175 targets** |
| target/control pairs passing the gate (attributed set) | m = 4: 172, m = 6: 155, m = 8: 149 |

The attribution set was cut from 220 to 120 videos to fit about 10 GPU-hours. The 120 are the first in the class-balanced sampling order.

## Gate: does reallocation leave the model unchanged?

500 videos, 343 targets, exact copies. Noisy copies agree to within about 1 pp.

| condition | agreement (%) | mean ΔP(pred) | target importance ratio I_t |
|---|---|---|---|
| original 16 distinct frames vs doubled reference | 94.1 | −0.001 | – |
| target, m = 4 / 6 / 8 | **93.6** / 87.5 / 85.4 | −0.020 / −0.067 / −0.110 | 0.98 / 1.01 / 1.04 |
| control, m = 4 / 6 / 8 | **93.3** / 87.2 / 81.9 | −0.028 / −0.077 / −0.127 | 0.80 / 0.57 / 0.42 |

- **m = 4 clears a 90% gate; m = 6 and m = 8 don't.**
- **The loss of agreement comes from the losers** (2 / 4 / 6 other contents cut to one copy), **not from duplicating the target.** The control, which duplicates an unimportant frame, loses as much or more.
- **The model's reliance on the target is unchanged** in the target condition (I_t ≈ 1.0).
- **Hence the per-pair gate rather than a single m cutoff.** It keeps m = 6 and 8 while comparing only clips with the reference's prediction.

## Results

### 1. Dilution slope β

β is the median slope of log(per-copy score ÷ reference) on log(m ÷ 2), with a bootstrap 95% CI. −1 = each copy gets 1/m; 0 = no dilution.

| method | target β, exact [95% CI] | target β, noise | control β, exact / noise | duplication-specific? |
|---|---|---|---|---|
| **Shapley (freeze)** | **−0.37** [−0.40, −0.33] | −0.37 | −0.01 / −0.01 | **yes** |
| **Integrated Gradients** | **−0.66** [−0.81, −0.51] | −0.70 | +0.02 / +0.08 | **yes** |
| **Leave-one-out (freeze)** | **−1.38** [−1.91, −1.07] | −1.51 | −0.11 / −0.13 | **yes (collapse)** |
| Occlusion | −0.04 [−0.19, 0.13] | −0.06 | +0.24 / +0.29 | no |
| Grad-CAM | −0.19 [−0.31, −0.12] | −0.15 | −0.15 / −0.15 | no (target ≈ control) |

n per row: Shapley 343 (targets × seeds), Grad-CAM 175, IG 137–139, LOO 134–136, occlusion 130. β uses only targets whose reference score is positive.

### 2. Per-copy and total credit

Exact copies, medians, each relative to the same method's score on the reference.

| method | per-copy score, m = 4 / 6 / 8 | total across copies, m = 4 / 6 / 8 |
|---|---|---|
| exact 1/m split, for comparison | 0.50 / 0.33 / 0.25 | 1.00 / 1.00 / 1.00 |
| **model's reliance (I_t ratio)** | – | **0.95 / 1.18 / 1.28** |
| Shapley | 0.76 / 0.69 / 0.62 | **1.52 / 2.07 / 2.47** |
| IG | 0.54 / 0.33 / 0.27 | 1.09 / 0.98 / 1.08 |
| LOO | **0.05 / −0.02 / −0.05** | **0.10 / −0.05 / −0.19** |
| occlusion | 0.69 / 0.60 / 0.47 | 1.37 / 1.81 / 1.87 |
| Grad-CAM | 0.91 / 0.80 / 0.74 | **1.83 / 2.38 / 2.94** |

The I_t row is the model's own reliance on the target. It's computed from the gate data, but uses the median over attributed pairs.

### 3. Ranking metrics

Exact copies.
- **Validated inversion:** the share of frames the method ranked clearly below the target at the reference that it ranks above the target in the condition, counted only where the model still ranks the target higher.
- **Top-1 loss:** cases where the target was the method's top frame at the reference but no longer is in the condition.

| method | validated inversion, target, m = 4 / 6 / 8 | same, control | top-1 loss, target (n), m = 4 / 6 / 8 | noise floor |
|---|---|---|---|---|
| Shapley | 7.9 / 11.6 / 15.9 % | 3.8 / 7.7 / 12.5 % | 0.63 / 0.66 / 0.71 (35 / 32 / 34) | 0.46 / 0.50 / 0.47 |
| IG | 14.2 / 18.1 / 12.2 % | 27.9 / 26.1 / 29.5 % | 0.57 / 0.58 / 0.52 (28 / 24 / 25) | 0 |
| LOO | **30.8 / 34.7 / 33.4 %** | 21.5 / 24.6 / 17.4 % | **0.84 / 0.88 / 0.94** (51 / 49 / 47) | 0 |
| occlusion | 16.4 / 19.7 / 15.9 % | 20.6 / 21.5 / 24.7 % | 0.34 / 0.36 / 0.46 (32 / 28 / 26) | 0 |
| Grad-CAM | 16.2 / 13.4 / 20.0 % | 22.4 / 34.7 / 27.6 % | 0.27 / 0.33 / 0.37 (22 / 18 / 19) | 0 |

- **Only Shapley and LOO show more inversions for the target than for the control.**
- **Top-1 loss rests on 18–51 cases per cell.** For Shapley, it's barely above its seed-noise floor.
- **Treat the ranking metrics as secondary evidence.**

## Drop-based methods and Play Fair

Here removed frames are **deleted**, so the model is evaluated on inputs of 1–16 frames. Two measurements on the same 500 test videos show how R3D's output depends on input length.

**Accuracy against input length (`eval_frame_count.py`).** The video is resampled to k frames, so content and length change together. Accuracy is normal only at **9–16 frames**: 36.4% at 8 frames against 64.8% at 9, and 15.6% at 1 frame. At ≤ 8 frames, layer3 already reduces time to a single position.

**Accuracy at fixed content (`eval_length_fixed_content.py`).** For each k, the same k distinct frames are evaluated in two ways:
- as a k-frame input (*native*);
- each repeated in consecutive slots to form a 16-frame input (*repeated*).

| k distinct frames | 1 | 2 | 4 | 6 | 8 | 9 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy, k-frame input (%) | 15.6 | 17.4 | 25.6 | 35.0 | 36.4 | 64.8 | 66.8 | 71.8 | 74.4 | 79.4 |
| accuracy, same frames repeated to 16 (%) | 63.8 | 72.8 | 78.4 | 77.6 | 77.6 | 79.0 | 79.8 | 79.8 | 79.2 | 79.4 |
| same prediction, native vs repeated (%) | 17.6 | 20.0 | 29.0 | 37.0 | 39.4 | 73.6 | 75.4 | 80.8 | 87.6 | 100 |

What the two measurements show:
- **The accuracy lost with fewer input frames is almost entirely an effect of input length, not of information content.** With the content fixed, the 16-frame presentation is 41–52 pp more accurate than the k-frame presentation for k ≤ 8, and 2–14 pp more accurate for k = 9–15.
- **The step between 8 and 9 frames is caused by the architecture.** It is present in the k-frame inputs (+28.4 pp) and absent in the repeated inputs (77.6% → 79.0%).
- **R3D's classifications depend mainly on the appearance of individual frames.** A single frame repeated 16 times is classified correctly in 63.8% of videos, and from k = 3 distinct frames on, accuracy is within 3 pp of the 16-frame value.

**Consequence.** Shapley-type methods weight every subset size equally, so about half the weight of Shapley-drop and Play Fair falls on subsets of ≤ 8 frames. R3D classifies those inputs 41–52 pp worse than the same frames presented at full length. The marginal contribution of a copy under deletion therefore contains a large component that depends on input length and not on the copy's content. The freeze-based methods evaluate 16-frame inputs of the *repeated* type, in which this component is absent.

**Settings:**
- **Shapley-drop:** 64 orderings × 2 seeds, 120 videos.
- **LOO-drop:** 120 videos.
- **Play Fair:** the official attributor via `run_playfair.py`, constructive sampler with **256** samples per size, **seed 0 only**, first **20** videos (27 targets).

**Play Fair and Shapley-drop agree within sampling noise.** Both compute Shapley values of the same value function: frames removed by deletion, softmax probability of the predicted class, uniform class prior for the empty set. They differ only in the Monte Carlo sampler (random orderings vs a fixed number of subsets per size), and are therefore unbiased estimators of the same Element Shapley Values. To test whether their estimates differ by more than sampling error, Play Fair (seed 0) is compared with Shapley-drop seed 0 **on identical conditions and targets**. Shapley-drop seed 1 vs seed 0 serves as the noise reference: those two differ only by sampling. Computed with `motivation/e1_compare_estimators.py`.

*Per-slot agreement* (164 shared conditions per copy type; near-duplicates agree within 0.002):

| | Play Fair vs Shapley-drop seed 0 | Shapley-drop seed 1 vs seed 0 (noise reference) |
|---|---|---|
| median correlation of slot attributions | **0.790** | 0.648 |
| median relative RMS difference ‖a − b‖ / ‖b‖ | **0.252** | 0.372 |
| conditions where Play Fair is closer to seed 0 than seed 1 is | **91%** | – |

*E1 metrics, paired on the same 27 targets* (target condition, exact copies; near-duplicates agree within 0.01):

| quantity | median paired difference, Play Fair − seed 0 [95% CI] | median paired difference, seed 1 − seed 0 [95% CI] | median absolute paired difference: Play Fair vs seed 0 / seed 1 vs seed 0 |
|---|---|---|---|
| per-copy ratio, m = 4 | −0.015 [−0.113, +0.006] | −0.040 [−0.066, +0.052] | 0.050 / 0.084 |
| per-copy ratio, m = 6 | −0.045 [−0.095, +0.045] | −0.038 [−0.116, +0.032] | 0.095 / 0.117 |
| per-copy ratio, m = 8 | −0.046 [−0.130, +0.019] | −0.054 [−0.165, +0.074] | 0.066 / 0.140 |
| total ratio, m = 4 | −0.030 [−0.226, +0.013] | −0.080 [−0.131, +0.104] | 0.099 / 0.168 |
| total ratio, m = 6 | −0.134 [−0.285, +0.136] | −0.113 [−0.349, +0.096] | 0.285 / 0.351 |
| total ratio, m = 8 | −0.184 [−0.518, +0.075] | −0.215 [−0.661, +0.296] | 0.263 / 0.562 |
| β | −0.059 [−0.116, +0.028] | −0.067 [−0.126, +0.046] | 0.082 / 0.113 |

What the comparison shows:
- **No systematic difference.** Every 95% confidence interval for the paired difference Play Fair − Shapley-drop contains 0, and each median paired difference is about the same as the difference between Shapley-drop's two seeds.
- **Play Fair is closer to Shapley-drop than Shapley-drop is to itself.** This holds per slot (in 91% of conditions) and per target, for every metric. That's expected: Play Fair's run used more model evaluations per condition (256 subsets per size, about 2,650 evaluations) than one Shapley-drop seed (64 orderings, about 1,000), so its estimates have less sampling error.
- **The unpaired medians mislead.** On these 27 targets, the median β is −0.206 for Play Fair, −0.008 for Shapley-drop seed 0 and −0.044 for seed 1. With few targets whose values are widely spread, the sample median moves considerably under small per-target differences, so the gap between medians (≈ 0.2) is much larger than the typical per-target difference (median paired difference −0.059). The same applies to the per-copy ratio at m = 8 (medians 0.769 vs 0.954; median paired difference −0.046). Estimators should therefore be compared with paired differences, not with differences between medians.

| method | target β, exact [95% CI] | control β | per-copy, m = 4 / 6 / 8 | total across copies, m = 4 / 6 / 8 |
|---|---|---|---|---|
| Shapley-freeze (for comparison) | −0.37 [−0.40, −0.33] | −0.01 | 0.76 / 0.69 / 0.62 | 1.52 / 2.07 / 2.47 |
| **Shapley-drop** | **−0.14** [−0.21, −0.09] | +0.02 | 0.89 / 0.87 / 0.84 | **1.79 / 2.60 / 3.38** |
| **Play Fair** (20 videos) | **−0.21** [−0.27, −0.02] | +0.11 | 0.96 / 0.89 / 0.77 | **1.93 / 2.68 / 3.08** |
| LOO-freeze (for comparison) | −1.38 [−1.91, −1.07] | −0.11 | 0.05 / −0.02 / −0.05 | 0.10 / −0.05 / −0.19 |
| **LOO-drop** | **−0.27** [−0.72, −0.15] | +0.17 | 0.33 / 0.16 / 0.01 | 0.65 / 0.49 / 0.02 |

Noisy copies give the same β to within 0.07. The model's reliance on the target (I_t ratio) is 0.95 / 1.18 / 1.28. The Play Fair row covers the first 20 videos only, while the Shapley-drop row covers all 120, so the two rows' medians aren't computed on the same targets; the paired comparison above is the like-for-like comparison.

**Reading:**
- **Shapley-drop and Play Fair barely dilute each copy** (0.77–0.96), so **the duplicated content's total credit grows about in proportion to m**: about 3× at m = 8, against the model's 1.28×. The control stays at about 1.0–1.2. With frames dropped, both treat copies as nearly independent contributors and **over-credit duplicated content**.
- **LOO-drop still collapses by m = 8** (per copy 0.01), but more slowly than LOO-freeze.
- **For R3D, these drop-based results are confounded by the model's dependence on input length.** Under deletion, the value function v(S) is evaluated on an input of |S| frames. Adding a copy of t to a subset S therefore changes both the content (no new information, since t is already in S) and the input length (|S| + 1 frames). R3D's accuracy is 36.4% with 8 frames and 64.8% with 9, because the temporal resolution of its intermediate layers changes at that boundary. The marginal contribution v(S ∪ {i}) − v(S) of a copy therefore includes a length-dependent component that is unrelated to the copy's content. That component is absent under freeze, where every input has 16 frames. The same component affects leave-one-out-drop: every single-slot removal shortens the input to 15 frames, which shifts all slot attributions by a similar amount and partly hides the decrease for duplicated content. The R3D conclusions are therefore based on the freeze-based and gradient-based methods. Play Fair's behaviour under duplication should be assessed on a model whose output doesn't depend on input length, such as V-JEPA2.

## Conclusions

**1. Duplication changes the attribution of a frame although the model's dependence on that frame is unchanged.**

Under duplication, R3D keeps the reference prediction (the gate retains 79–91% of target–control pairs). The model-based importance of the duplicated content t, measured by removing all of its copies, is 0.95, 1.18 and 1.28 times its reference value at m = 4, 6 and 8. The model's dependence on t is therefore approximately unchanged at m = 4 and slightly higher at m = 6 and 8. Nevertheless, for Shapley (freeze), Integrated Gradients and leave-one-out (freeze), the attribution assigned to each copy of t decreases with m. In the matched control conditions, where an unimportant content is duplicated instead and t keeps two copies, the attribution of t is unchanged (control β between −0.11 and +0.02). The decrease is therefore caused by the duplication of t, not by the accompanying structural changes to the clip.

**2. The methods differ in how duplication changes the attribution.**

For each method, the per-copy ratio (mean attribution of one copy of t, relative to the reference) and the total ratio (summed attribution of all copies, relative to the reference) are compared with two reference cases:
- **exact division of a fixed total:** per-copy ratio 2/m (0.50, 0.33, 0.25), total ratio 1;
- **no change per copy:** per-copy ratio 1, total ratio m/2 (2, 3, 4).

- **Leave-one-out (freeze) assigns approximately zero attribution to each copy.** The per-copy ratio is 0.05 at m = 4 and indistinguishable from zero at m = 6 and 8. When one copy is removed, the remaining copies are still present in the input, so the model's output is almost unchanged and the marginal effect of each copy is close to zero. A content on which the model depends is therefore reported as unimportant. Leave-one-out also has the highest rate of validated rank inversions (31–35%). In 84–94% of cases where t was the highest-ranked content in the reference, it is no longer highest-ranked under duplication.
- **Integrated Gradients divides a fixed total among the copies.** The per-copy ratio (0.54, 0.33, 0.27) closely follows 2/m, and the total ratio stays between 0.98 and 1.09. This follows from the completeness property of Integrated Gradients: the attributions sum to the difference between the model's output on the input and on the baseline. Each copy of t is therefore reported as approximately m/2 times less important than in the reference.
- **Shapley (freeze) partially reduces the attribution of each copy and increases the total.** The per-copy ratio decreases to 0.62 at m = 8, which is below 1 but well above 2/m (0.25). The total ratio increases to 2.47, which exceeds the model's dependence on t (I_t ratio 1.28). Each copy is therefore reported as less important than in the reference, and the content as a whole as more important than the model's behaviour supports.
- **Grad-CAM increases the total attribution of the content.** Each copy keeps most of its reference attribution (per-copy ratio 0.74 at m = 8), so the total ratio increases to 2.94. The small decrease per copy also occurs in the control condition (control β −0.15, target β −0.19), so it isn't caused by the duplication of t.
- **Occlusion shows no change specific to duplication.** The target β (−0.04) doesn't differ from zero, and the control condition changes by a similar amount.

**3. None of the methods is consistent with the model on both measures.**

A method consistent with the model would leave the attribution of t unchanged when the model's dependence on t is unchanged. In every method tested, either the per-copy attribution falls below the reference value (leave-one-out, Integrated Gradients, Shapley) or the total attribution rises above the model's dependence (Shapley, Grad-CAM). Consequently, when a video contains repeated or near-identical frames, the attribution assigned to a content depends on how many times it occurs, and not only on how much the model depends on it. Natural videos frequently contain such repetition: static shots, slow motion, and densely sampled clips.

**4. The effects don't depend on the copies being pixel-identical.**

With near-duplicates (Gaussian noise, σ = 2/255), the dilution slopes agree with those for exact copies to within 0.15 for every method. The effects therefore aren't an artefact of identical inputs.

**5. Drop-based methods, including Play Fair, can't be evaluated reliably on R3D.**

Play Fair and Shapley-drop compute Shapley values of the same value function, and their estimates agree within sampling noise: on identical targets, the paired differences between them are no larger than those between two seeds of Shapley-drop, with no systematic shift (median paired difference in β −0.059 [−0.116, +0.028], against −0.067 between seeds). Under deletion, however, R3D's output depends strongly on input length when the content is held fixed: the same k ≤ 8 frames are classified correctly in 25–36% of videos as a k-frame input, and in 73–80% as a 16-frame input with repeated frames. The resulting attributions therefore combine the effect of the frames' content with the effect of input length. Their behaviour under duplication (per-copy ratio 0.77–0.96, total ratio about 3 at m = 8) can't be attributed to Play Fair alone, and should be evaluated on a model whose output doesn't depend on input length.

## Limitations

- **Only one model.** VideoMAE, V-JEPA2 and the two TRNs are still to come.
- **Play Fair was run on 20 videos with reduced settings.** For R3D, all drop-based results (Play Fair, Shapley-drop, LOO-drop) are confounded by R3D's 8/9-frame cliff (see above).
- **Selection:**
  - Only correctly classified videos with at least one important frame (I > 0.01) are included, which is 44% of those sampled.
  - The pair gate keeps fewer pairs at higher m (m = 8: 149 vs m = 4: 172), favouring videos that cope with losing other frames. At m = 4 alone, the per-copy pattern is the same: Shapley 0.76, IG 0.54, LOO 0.05.
- **The reference isn't the original clip.** It shows 8 frames twice each. It agrees with the original on 94% of videos, and confidence changes by only −0.001.
- **The control isn't fully neutral at high m.** The target's importance in the control falls to 0.42–0.80, so read results as the gap between target and control.
- **Sample size:** 120 videos attributed, not the planned 300. CIs are tight for Shapley and IG, wider for LOO (but entirely below −1).
- **β for LOO:** values below −1 mostly reflect scores close to 0, not a meaningful rate. Use the per-copy ratio and the collapse framing instead.

## Pending

1. **Play Fair on all 120 videos** (optional for R3D, given the length confound). The full-default setting (1024 samples, 2 seeds) is about 30 h; 256 samples and 1 seed is about 4 h.
   ```
   python motivation/e1_attribute.py --model r3d --limit 120 --methods playfair --seeds 0 --pf-max-samples 256
   python motivation/e1_metrics.py attr --model r3d
   ```
2. **Figures** for these results.
3. **The same experiment on VideoMAE, the two TRNs and V-JEPA2.**

## Files (`results/e1/`)

| file | contents |
|---|---|
| `r3d_conditions.jsonl`, `r3d_skipped.jsonl` | 500-video conditions (m = 4, 6, 8) and skipped videos |
| `r3d_gate.csv` | gate report |
| `r3d_attributions.jsonl` | per-slot scores for every attributed condition, method and seed |
| `r3d_rows.csv`, `r3d_slopes.csv`, `r3d_summary.csv`, `r3d_beta.csv` | metrics: per row, per-target slopes, per method × m, and β with CIs |
| `r3d_build.log`, `r3d_attribute.log` | run logs |
| `../frame_count_r3d_*.csv`, `../frame_count_r3d_short_*.csv` | accuracy against input length (uniform resampling), k = 1–24 |
| `../length_fixed_content_r3d_*.csv` | accuracy at fixed content: k-frame vs repeated-to-16 inputs, k = 1–16 |
| `r3d_insert_pilot/`, `r3d_realloc_pilot/`, `r3d_realloc500_m48/` | superseded earlier runs |

Reproduce:
```
python motivation/e1_build_conditions.py --model r3d --limit 500 --ms 2 4 6 8
python motivation/e1_metrics.py gate --model r3d
python motivation/e1_attribute.py --model r3d --limit 120
python motivation/e1_attribute.py --model r3d --limit 120 --methods shapley_drop loo_drop
python motivation/e1_attribute.py --model r3d --limit 20 --methods playfair --seeds 0 --pf-max-samples 256
python motivation/e1_metrics.py attr --model r3d

# Play Fair vs Shapley-drop, paired on identical conditions (seed 1 of Shapley-drop as noise reference)
python motivation/e1_compare_estimators.py --model r3d

# R3D accuracy at 1-11 frames (the 8/9-frame cliff)
python eval_frame_count.py --models r3d --frames 1 2 3 4 5 6 7 8 9 10 11 --limit 500 --datasets r3d=ucf101_test --out results/frame_count_r3d_short

# R3D accuracy at fixed content: k frames as a k-frame input vs repeated to 16 frames
python eval_length_fixed_content.py --models r3d --datasets r3d=ucf101_test --ks 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --limit 500 --out results/length_fixed_content_r3d
```
