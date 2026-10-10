# E1 summary: MC3-18 on UCF101

## In short

MC3-18 is the one UCF101 model in our set whose accuracy doesn't depend on clip length. So on this model we can test the methods that **delete** frames, including Play Fair, without the length problem that R3D has.

Unlike the R3D models, MC3-18 **does** change when a frame is duplicated. It relies on the duplicated frame more and more: 2.0 times as much at m = 3, and 3.2 times at m = 9. This follows from its design, which averages over all frames, so every copy adds weight.

Measured against that:
- **Shapley, Play Fair, Integrated Gradients and Grad-CAM give the copies too much credit in total**, 4.3 to 4.9 times the reference at m = 9, against the model's 3.2.
- **Leave-one-out gives them far too little** (1.2 times). Each copy keeps only 13% of its credit at m = 9.
- **Occlusion gives them a little too little** (2.5 times).
- **Play Fair gives the same values as Shapley-drop**, within the noise between two Shapley-drop seeds.

## 1. The question

When a frame the model relies on appears m times in a clip, does an attribution method still give it the right amount of credit?

"The right amount" is judged against the model, as for the other models. We measure how much the model relies on the frame before and after duplication: the I_t ratio. On MC3-18 this ratio rises with m. So here the copies' total credit *should* grow, by as much as the I_t ratio.

## 2. How the test works

**Model.** MC3-18 from torchvision (mixed convolutions: 3D in the first stage, spatial only after that), pretrained on Kinetics-400 and fine-tuned on UCF101 split 1, published as `dronefreak/mc3-18-ucf101` (wrapper: `models/torchvision_ucf101.py`). It takes 16 frames at 112 px. Its test accuracy is 86.83% on the full split-1 test set (reported: 87.05%).

**Why insertion, with deleted frames.** MC3-18 never downsamples time after its first stage, and averages over all time positions at the end. Its accuracy is flat from 3 to 24 frames (87.6–88.6% at 17–24 frames, against 88.0% at 16). Shown the same frames as a short clip or repeated to 16 frames, its accuracy differs by at most 0.4 points. So adding copies (the clip gets longer) and deleting frames (the clip gets shorter) are both safe. See `results/design_choice_report.md`, section 7.

**Clips** (the insertion design, as for V-JEPA2):
- **Reference:** the original 16 frames, each once (m = 1).
- **Target clip:** m − 1 extra copies of the target are inserted next to it. The clip gets longer: 18, 20 or 24 frames for m = 3, 5 and 9.
- **Control clip (spread control):** m − 1 other frames, chosen at random, each get one extra copy next to them. The target stays single. So the control has the same length as the target clip, and the same number of duplicated frames, but no frame has more than two copies.

Exact copies, and copies with a little noise (σ = 2/255).

**Targets.** Chosen by the model's own importance I(c), never by an attribution method. One "high" target from the top 2 frames, and one "mid" target from ranks 3–8 when available. Both need I > 0.01.

**Methods.**
- Shapley-drop: deleted frames, 64 orderings, 2 seeds;
- Play Fair: the authors' own code, 256 samples per subset size, 1 seed, the first 20 videos;
- leave-one-out (drop);
- occlusion (one frame replaced by a blank frame);
- Integrated Gradients (32 steps, blank baseline);
- Grad-CAM (on `layer3`).

## 3. Which videos were used

| step | number |
|---|---|
| test videos screened | 500 |
| left out: the model got them wrong | 71 |
| left out: no single frame mattered enough (I ≤ 0.01) | 164 |
| **kept and attributed** | **265 videos, 417 targets** |
| target–control pairs that passed the gate | 411 at m = 3, 410 at m = 5, 400 at m = 9 |
| Play Fair | the first 20 videos, 32 targets |

Many videos have no single important frame (164 of 500). MC3-18 recognises 78% of videos from one frame alone, so removing any one frame often changes nothing.

## 4. Does duplication change the model?

Checked on all 417 targets before running any attribution method. Exact copies; noisy copies are within 0.6 percentage points.

| clip | same prediction as the reference | change in the predicted class's probability | how much the model relies on the target (I_t ratio) |
|---|---|---|---|
| target clip, m = 3 / 5 / 9 | 99.5% / 99.5% / 98.8% | +0.024 / +0.037 / +0.051 | **2.01 / 2.55 / 3.19** |
| control clip, m = 3 / 5 / 9 | 98.8% / 98.8% / 96.9% | −0.009 / −0.016 / −0.029 | 0.90 / 0.83 / 0.70 |

What this shows:
- **The model keeps its prediction almost always** (97–100%). The per-pair gate keeps 96–99% of pairs.
- **The model relies on the duplicated target much more**: 2 to 3.2 times as much. Extra copies of the target also raise the predicted class's probability. This is what an average over time does: m copies give the target m times the weight in the average.
- **In the control clips, the model relies on the target a little less** (0.70 to 0.90), because the clip is longer and the target is a smaller part of it.

This is different from R3D-50 and R3D-18, where the model's reliance on the target stayed near 1.

## 5. Results

### 5.1 Credit per copy and credit in total

- **per-copy ratio:** the credit of one copy, divided by its credit in the reference;
- **total ratio:** the credit of all copies added up, divided by the target's credit in the reference.

Reference points (the reference has one copy):
- **The credit is split evenly between the copies:** each copy gets 1/m (0.33, 0.20, 0.11), and the total stays at 1.
- **The credit follows the model, shared evenly between the copies:** the total equals the I_t ratio, and each copy gets I_t / m (0.67, 0.52, 0.36).

Exact copies, medians over targets:

| method | per-copy ratio, m = 3 / 5 / 9 | total ratio, m = 3 / 5 / 9 |
|---|---|---|
| (credit split evenly) | 0.33 / 0.20 / 0.11 | 1.00 / 1.00 / 1.00 |
| **(credit follows the model: I_t ratio)** | **0.67 / 0.52 / 0.36** | **2.00 / 2.58 / 3.22** |
| Shapley-drop | 0.77 / 0.65 / 0.48 | 2.32 / 3.23 / 4.33 |
| Play Fair (20 videos; model on these: 2.06 / 2.72 / 3.34) | 0.81 / 0.68 / 0.48 | 2.43 / 3.39 / 4.32 |
| Integrated Gradients | 0.81 / 0.68 / 0.54 | 2.42 / 3.39 / 4.82 |
| Leave-one-out | 0.40 / 0.20 / 0.13 | 1.21 / 1.02 / 1.18 |
| Occlusion | 0.62 / 0.42 / 0.27 | 1.87 / 2.12 / 2.45 |
| Grad-CAM | 0.83 / 0.70 / 0.54 | 2.49 / 3.50 / 4.89 |

The I_t row uses only the attributed pairs, so it differs a little from section 4.

### 5.2 What each method does

- **Shapley-drop and Play Fair give the copies too much in total.** At m = 9, the copies get 4.3 times the reference credit, while the model relies on the target 3.2 times as much (Play Fair: 3.3 on its 20 videos). Each copy keeps 0.48 of its credit, which is more than the 0.36 that an even share of the model's reliance would give.
- **Integrated Gradients and Grad-CAM give even more in total** (4.8 and 4.9 at m = 9). On R3D-50, Integrated Gradients kept its total at 1, because its scores must add up to a fixed amount: the logit difference between the clip and a blank clip. On MC3-18 that logit difference itself grows with the copies, so the total can grow.
- **Leave-one-out gives far too little.** The total stays near 1 while the model's reliance triples. Each copy keeps 0.40, 0.20 and 0.13 of its credit, and in 9%, 17% and 25% of targets every copy's credit collapses to almost 0. Removing one copy leaves the others in the clip, so the output barely changes.
- **Occlusion gives a little too little** (2.45 against 3.22 at m = 9).

### 5.3 Is the change caused by duplicating the target?

The dilution slope β describes how fast the credit per copy falls as m grows: β = −1 means the credit is split evenly, β = 0 means each copy keeps its full credit.

On MC3-18 the control clips are longer too, so the target's credit falls a little in the control as well. That fall follows the model: in the control, Shapley's credit for the target (0.90 / 0.85 / 0.73) matches the model's I_t ratio (0.90 / 0.83 / 0.70). So compare the target's β with the control's β.

Two more reference points for β:
- **target, if the credit follows the model (I_t / m):** β ≈ −0.44;
- **control, if the credit follows the model:** β ≈ −0.14.

| method | β, target (exact) [95% range] | β, target (noisy) | β, control (exact / noisy) |
|---|---|---|---|
| (credit follows the model) | −0.44 | – | −0.14 |
| Shapley-drop | −0.31 [−0.32, −0.28] | −0.31 | −0.12 / −0.12 |
| Play Fair (20 videos) | −0.28 [−0.32, −0.24] | −0.28 | −0.12 / −0.12 |
| Integrated Gradients | −0.25 [−0.27, −0.23] | −0.25 | −0.16 / −0.16 |
| Leave-one-out | −0.77 [−0.84, −0.70] | −0.77 | −0.14 / −0.15 |
| Occlusion | −0.53 [−0.58, −0.49] | −0.53 | −0.16 / −0.16 |
| Grad-CAM | −0.25 [−0.26, −0.23] | −0.24 | −0.18 / −0.18 |

What this shows:
- **Every method's credit per copy falls faster for the target than for the control.** So duplication causes it.
- **In the control, every method follows the model** (β −0.12 to −0.18, against −0.14).
- **For the target, Shapley, Play Fair, Integrated Gradients and Grad-CAM fall more slowly than the model's reliance per copy** (−0.25 to −0.31, against −0.44). That's the same finding as their too-high totals in 5.1.
- **Leave-one-out falls much faster** (−0.77), and occlusion a little faster (−0.53).

Number of targets per row: Shapley-drop 824 (both seeds counted), Play Fair 32, the others 410–415.

### 5.4 Does duplication change the ranking of frames?

Exact copies:

| method | inversions, target (m = 3 / 5 / 9) | inversions, control | top-1 loss, target | top-1 loss from random noise alone |
|---|---|---|---|---|
| Shapley-drop | 2% / 2% / 1% | 5% / 7% / 6% | 0.60 / 0.49 / 0.24 | 0.62 |
| Play Fair | 1% / 3% / 3% | 1% / 1% / 0% | 0.23 / 0.46 / 0.38 | (1 seed only) |
| Integrated Gradients | 1% / 2% / 1% | 5% / 10% / 14% | 0.28 / 0.31 / 0.28 | 0 |
| Leave-one-out | 13% / 14% / 12% | 0% / 0% / 0% | 0.56 / 0.67 / 0.68 | 0 |
| Occlusion | 7% / 8% / 7% | 2% / 4% / 7% | 0.39 / 0.46 / 0.43 | 0 |
| Grad-CAM | 7% / 12% / 18% | 2% / 3% / 6% | 0.33 / 0.49 / 0.63 | 0 |

- **Shapley, Play Fair and Integrated Gradients keep the target on top.** The model relies on the target more and more, and these methods follow. Shapley's top-1 loss (0.24–0.60) is no higher than the noise between its two seeds (0.62).
- **Leave-one-out and Grad-CAM do change the ranking.** Leave-one-out has 12–14% inversions for the target, against 0% for the control. Grad-CAM's top-1 loss rises to 0.63 at m = 9.

## 6. Play Fair gives the same values as Shapley-drop

Play Fair and Shapley-drop compute the same quantity: the model's probability for the predicted class, with deleted frames, and a uniform class prior for an empty clip. They differ only in how they pick random subsets of frames. On R3D-50 they agreed within noise. MC3-18 repeats the check on a model where deleting frames is valid.

Same 32 targets. As a measure of pure noise, we compare Shapley-drop with itself, using two seeds.

| | Play Fair vs Shapley-drop (seed 0) | Shapley-drop seed 1 vs seed 0 (noise only) |
|---|---|---|
| correlation of the per-frame scores (median) | 0.841 | 0.775 |
| relative difference ‖a − b‖ / ‖b‖ (median) | 0.423 | 0.503 |
| conditions where Play Fair is closer to seed 0 than seed 1 is | 88% | – |

Per target (exact copies; noisy copies within 0.01):

| measure | median difference, Play Fair − seed 0 [95% range] | median difference, seed 1 − seed 0 [95% range] | typical size of the difference: Play Fair vs seed 0 / seed 1 vs seed 0 |
|---|---|---|---|
| per-copy ratio, m = 3 | +0.005 [−0.081, +0.082] | +0.023 [−0.063, +0.158] | 0.117 / 0.212 |
| per-copy ratio, m = 5 | +0.097 [−0.043, +0.215] | +0.051 [−0.088, +0.237] | 0.229 / 0.288 |
| per-copy ratio, m = 9 | +0.035 [−0.087, +0.101] | +0.029 [−0.084, +0.096] | 0.126 / 0.145 |
| total ratio, m = 3 | +0.016 [−0.244, +0.245] | +0.069 [−0.188, +0.474] | 0.351 / 0.636 |
| total ratio, m = 5 | +0.485 [−0.215, +1.075] | +0.257 [−0.439, +1.185] | 1.144 / 1.438 |
| total ratio, m = 9 | +0.315 [−0.786, +0.913] | +0.265 [−0.755, +0.865] | 1.137 / 1.302 |
| β | +0.085 [−0.077, +0.185] | +0.058 [−0.069, +0.176] | 0.187 / 0.229 |

What this shows:
- **There's no systematic difference.** Every 95% range includes 0, and the differences are no larger than the noise between Shapley-drop's two seeds.
- **Play Fair is closer to Shapley-drop than Shapley-drop is to itself.** That's expected: Play Fair used more model evaluations per clip (3,090 to 5,418, against about 1,100 for one Shapley-drop seed).
- **So the Shapley-drop results stand for Play Fair**, on all 265 videos.

## 7. Conclusions

1. **On MC3-18, duplication changes the model.** It relies on the target 2 to 3.2 times as much, because it averages over time. This makes MC3-18 a different test from the R3D models, where the model's reliance stayed near 1.

2. **Most methods overshoot the model's change.** Shapley-drop, Play Fair, Integrated Gradients and Grad-CAM give the copies 4.3 to 4.9 times the reference credit at m = 9, against the model's 3.2. Their credit per copy falls, but more slowly than the model's reliance per copy.

3. **Leave-one-out undershoots badly.** The copies' total stays near 1 while the model's reliance triples, and many targets lose almost all credit. It also changes the ranking: 12–14% inversions, against 0% in the control.

4. **The same pattern as on the R3D models, from a different direction.** On all three UCF101 models, no method's total matches the model: Shapley-type methods and Grad-CAM give too much, leave-one-out too little. On MC3-18 the model's own reliance rises, so the per-copy fall is partly correct here. The error is in how much.

5. **Play Fair and Shapley-drop agree within noise** on MC3-18, as on R3D-50. Because deleting frames is valid on MC3-18, this is the cleanest evidence that our Shapley-drop results apply to Play Fair.

6. **Noisy copies give the same results as exact copies** (β within 0.01).

## 8. Limitations

- **Selection.** 265 of 500 videos (53%) were kept. 164 were left out because no single frame mattered: MC3-18 recognises most videos from a single frame (78% accuracy from one frame).
- **Sampling noise in Shapley-drop.** Its scores from two seeds correlate at 0.78 per frame. That's why the top-1 loss from noise alone is high (0.62), and why top-1 loss can't show a Shapley effect here.
- **The same seed for every video.** E1 uses the same random orderings for every clip of the same length. The noise is independent between a target clip and its reference (they have different lengths), so the ratios aren't biased. But the noise is shared across videos, so the 95% ranges may be somewhat too narrow. (The E2/E3 code uses a different seed per video.)
- **Play Fair** ran on 20 videos, with 256 samples per subset size and 1 seed, the same settings as for R3D-50.
- **Confidence still depends a little on length.** With few distinct frames, a short clip gets a higher probability for the true class than the same frames repeated to 16 (0.74 against 0.57 for one frame). This adds a small length effect to the methods that delete frames. It's much smaller than R3D's, and it doesn't change accuracy.
- **Two machines.** The clips and gate checks were computed on a laptop, and the attribution in fp16 on Colab. The reference scores are recomputed in the attribution run, so every ratio compares scores from the same run.

## 9. Files and how to reproduce

Files in `results/e1/`:

| file | what it contains |
|---|---|
| `mc3_18_conditions.jsonl`, `mc3_18_skipped.jsonl` | the clips built for each video (m = 3, 5, 9), and the videos left out |
| `mc3_18_gate.csv` | the gate check (section 4) |
| `mc3_18_attributions.jsonl` | the per-frame scores for every clip, method and seed, including Play Fair |
| `mc3_18_rows.csv`, `mc3_18_slopes.csv`, `mc3_18_summary.csv`, `mc3_18_beta.csv` | the measures, and β with 95% ranges |
| `mc3_18_compare_estimators.log` | Play Fair vs Shapley-drop (section 6) |
| `mc3_18_build.log`, `mc3_18_metrics.log` | run logs |
| `../frame_count_tv_*.csv`, `../length_fixed_content_tv_*.csv` | accuracy against clip length, and at fixed content |

Commands:
```
python motivation/e1_build_conditions.py --model mc3_18 --limit 500
python motivation/e1_metrics.py gate --model mc3_18
python motivation/e1_attribute.py --model mc3_18 --fp16 --methods shapley_drop loo_drop occlusion ig gradcam
python motivation/e1_attribute.py --model mc3_18 --limit 20 --fp16 --methods playfair --seeds 0 --pf-max-samples 256
python motivation/e1_metrics.py attr --model mc3_18
python motivation/e1_compare_estimators.py --model mc3_18
```
