# E1 summary: R3D-50 on UCF101

## In short

We tested what attribution methods do when an important frame appears several times in a clip.

The model (R3D-50) barely changes its behaviour when a frame is duplicated. It keeps its prediction, and it relies on the duplicated frame about as much as before.

The attribution methods do change:
- **Leave-one-out** gives each copy almost no credit.
- **Integrated Gradients** splits the credit evenly between the copies.
- **Shapley** gives each copy less credit, but gives the copies more credit in total.
- **Grad-CAM** gives the copies more credit in total.
- **Occlusion** shows no clear effect of duplication.

So no method gives a duplicated frame the credit that matches how the model uses it.

## 1. The question

When a frame the model relies on appears m times in a clip, does an attribution method still give it the right amount of credit?

"The right amount" is judged against the model itself. If the model relies on the frame as much as before, the frame's credit shouldn't change just because it's repeated.

## 2. How the test works

**Model.** R3D-50 (a 3D ResNet-50), fine-tuned on UCF101. It takes 16 frames. Checkpoint: `models/r3d/ckpt/save_200.pth`.

**Data.** The UCF101 split-1 test set, with classes sampled evenly.

**Clips.** We use the *reallocation* design. The clip always has 16 slots:
- We pick 8 frames from the video (every other one of 16 sampled frames). We call them *contents*.
- **Reference clip:** each content appears twice, in two neighbouring slots. So the reference is the m = 2 level.
- **Target clip:** one important content, the *target* t, gets m slots. To make room, some other contents (the *losers*) drop from two copies to one.
- **Control clip:** the same losers drop to one copy, but the freed slots go to the least important content (the *recipient*). The target keeps its two copies.

We test m = 4, 6 and 8. R3D can't use the simpler *insertion* design (adding copies, so the clip gets longer), because its accuracy drops when the clip is longer than 16 frames. See `results/design_choice_report.md`.

**Copy types.** Each copy is either an exact duplicate, or a duplicate with a little random noise added (σ = 2/255).

**Choosing the target.** We measure how much the model needs each content. We remove every copy of it and see how much the predicted class's probability drops. This is the content's *importance* I(c). The target is one of the most important contents. A second, "mid" target is chosen from ranks 3–4 when available. Both need I > 0.01. No attribution method is used to choose them.

**Removing frames.** Some methods need to remove frames. Here, a removed slot is filled with a neighbouring frame (`late` freeze). So the clip always stays at 16 frames.

**Methods tested:**
- Shapley (freeze), 64 random orderings, 2 seeds;
- leave-one-out (freeze);
- occlusion (one frame replaced by a blank frame);
- Integrated Gradients (32 steps, blank baseline);
- Grad-CAM (on `layer2`).

## 3. Which videos were used

| step | number |
|---|---|
| test videos screened | 500 |
| left out: the model got them wrong | 102 |
| left out: no single content mattered enough (I ≤ 0.01) | 178 |
| kept | 220 videos, 343 targets |
| **used for attribution** (the first 120 kept videos) | **120 videos, 175 targets** |
| target–control pairs that passed the gate | 172 at m = 4, 155 at m = 6, 149 at m = 8 |

We attributed 120 of the 220 videos to keep the run to about 10 GPU-hours.

## 4. Does duplication change the model?

We checked this on all 500 videos (343 targets), before running any attribution method. Exact copies are shown. Noisy copies give values within about 1 percentage point.

| clip | same prediction as the reference | change in the predicted class's probability | how much the model relies on the target (I_t ratio) |
|---|---|---|---|
| original 16 frames vs the reference clip | 94.1% | −0.001 | – |
| target clip, m = 4 / 6 / 8 | 93.6% / 87.5% / 85.4% | −0.020 / −0.067 / −0.110 | 0.98 / 1.01 / 1.04 |
| control clip, m = 4 / 6 / 8 | 93.3% / 87.2% / 81.9% | −0.028 / −0.077 / −0.127 | 0.80 / 0.57 / 0.42 |

The I_t ratio compares how much the model relies on the target in the clip with how much it relied on it in the reference. A value of 1 means "the same".

What this shows:
- **The model relies on the duplicated target as much as before** (I_t ratio about 1.0).
- **The prediction changes more often at higher m, but not because of the target.** The control clips change just as often. The cause is the losers: 2, 4 or 6 other contents lose a copy, so the model sees less of the rest of the video.
- **Showing every content twice (the reference) is almost harmless.** It changes the prediction in only 5.9% of videos.

**Which clips we used: the per-pair gate.** We only compare clips that the model classifies like the reference. There are two ways to apply this:
- **Per m:** keep a whole level of m only if at least 90% of its target clips keep the prediction. This would keep only m = 4. *We didn't use this rule.*
- **Per pair:** check each target clip together with its control clip. Keep the pair only if both keep the prediction. Drop only the pairs that fail. *This is the rule we used.*

With the per-pair rule, all three levels are used: 91%, 82% and 79% of pairs are kept at m = 4, 6 and 8.

## 5. Results

### 5.1 How much credit each copy gets, and how much all copies get together

For each method we compare the target's credit with its credit in the reference clip:
- **per-copy ratio:** the credit of one copy, divided by its credit in the reference;
- **total ratio:** the credit of all copies added up, divided by the total in the reference.

Two reference points help to read the numbers:
- **The credit is split evenly between the copies:** each copy gets 2/m (0.50, 0.33, 0.25), and the total stays at 1.
- **Each copy keeps its full credit:** each copy gets 1, and the total grows to m/2 (2, 3, 4).

Exact copies, medians over targets:

| method | per-copy ratio, m = 4 / 6 / 8 | total ratio, m = 4 / 6 / 8 |
|---|---|---|
| (credit split evenly) | 0.50 / 0.33 / 0.25 | 1.00 / 1.00 / 1.00 |
| **(how much the model relies on t, I_t ratio)** | – | **0.95 / 1.18 / 1.28** |
| Shapley | 0.76 / 0.69 / 0.62 | 1.52 / 2.07 / 2.47 |
| Integrated Gradients | 0.54 / 0.33 / 0.27 | 1.09 / 0.98 / 1.08 |
| Leave-one-out | 0.05 / −0.02 / −0.05 | 0.10 / −0.05 / −0.19 |
| Occlusion | 0.69 / 0.60 / 0.47 | 1.37 / 1.81 / 1.87 |
| Grad-CAM | 0.91 / 0.80 / 0.74 | 1.83 / 2.38 / 2.94 |

The I_t row here uses only the pairs that were attributed, so it differs a little from the table in section 4.

### 5.2 What each method does

- **Leave-one-out gives each copy almost no credit.** Each copy gets 5% of its reference credit at m = 4, and about 0 at m = 6 and 8. This happens because removing one copy leaves the other copies in the clip, so the model's output hardly changes. The model needs this content, but the method reports it as unimportant.
- **Integrated Gradients splits the credit evenly.** Each copy gets about 2/m, and the total stays near 1. This follows from how the method works: its scores always add up to the same fixed total. So each copy looks m/2 times less important than in the reference.
- **Shapley lowers each copy's credit, but raises the total.** Each copy drops to 0.62 at m = 8. That's lower than in the reference, but much higher than an even split (0.25). Added up, the copies get 2.47 times the reference credit. The model relies on the target only 1.28 times as much. So each copy looks less important, and the content as a whole looks more important, than the model's behaviour supports.
- **Grad-CAM raises the total.** Each copy keeps most of its credit (0.74 at m = 8), so the total grows to 2.94. Its small drop per copy also appears in the control, so that drop isn't caused by duplication.
- **Occlusion shows no clear effect of duplication.** The target's credit and the control's credit change by similar amounts.

### 5.3 Is the change caused by duplicating the target?

To check, we compare the target clip with the control clip. We summarise each with one number, the **dilution slope β**. It describes how fast the credit per copy falls as m grows:
- β = −1: the credit is split evenly between the copies;
- β = 0: each copy keeps its full credit.

If β is clearly negative for the target, but near 0 for the control, the change is caused by duplicating the target.

| method | β, target (exact copies) [95% range] | β, target (noisy copies) | β, control (exact / noisy) | caused by duplication? |
|---|---|---|---|---|
| Shapley | −0.37 [−0.40, −0.33] | −0.37 | −0.01 / −0.01 | yes |
| Integrated Gradients | −0.66 [−0.81, −0.51] | −0.70 | +0.02 / +0.08 | yes |
| Leave-one-out | −1.38 [−1.91, −1.07] | −1.51 | −0.11 / −0.13 | yes (credit collapses) |
| Occlusion | −0.04 [−0.19, 0.13] | −0.06 | +0.24 / +0.29 | no |
| Grad-CAM | −0.19 [−0.31, −0.12] | −0.15 | −0.15 / −0.15 | no (target and control change alike) |

Notes:
- β only uses targets whose reference credit is positive. Number of targets per row: Shapley 343 (both seeds counted), Grad-CAM 175, Integrated Gradients 137–139, leave-one-out 134–136, occlusion 130.
- For leave-one-out, β below −1 just means the credit went to almost 0. The exact value isn't meaningful. The per-copy ratio describes it better.

### 5.4 Does duplication change the ranking of frames?

These numbers are less reliable, so we treat them as extra evidence only.
- **Inversion:** a frame that the method ranked clearly below the target in the reference is ranked above it after duplication. We count only cases where the model itself still ranks the target higher.
- **Top-1 loss:** the target was the method's top frame in the reference, but isn't any more.

Exact copies:

| method | inversions, target (m = 4 / 6 / 8) | inversions, control | top-1 loss, target (number of cases) | top-1 loss from random noise alone |
|---|---|---|---|---|
| Shapley | 7.9% / 11.6% / 15.9% | 3.8% / 7.7% / 12.5% | 0.63 / 0.66 / 0.71 (35 / 32 / 34) | 0.46 / 0.50 / 0.47 |
| Integrated Gradients | 14.2% / 18.1% / 12.2% | 27.9% / 26.1% / 29.5% | 0.57 / 0.58 / 0.52 (28 / 24 / 25) | 0 |
| Leave-one-out | 30.8% / 34.7% / 33.4% | 21.5% / 24.6% / 17.4% | 0.84 / 0.88 / 0.94 (51 / 49 / 47) | 0 |
| Occlusion | 16.4% / 19.7% / 15.9% | 20.6% / 21.5% / 24.7% | 0.34 / 0.36 / 0.46 (32 / 28 / 26) | 0 |
| Grad-CAM | 16.2% / 13.4% / 20.0% | 22.4% / 34.7% / 27.6% | 0.27 / 0.33 / 0.37 (22 / 18 / 19) | 0 |

- Only Shapley and leave-one-out have more inversions for the target than for the control.
- Top-1 loss is based on only 18–51 cases. For Shapley it's only a little above what random noise alone produces.

## 6. Methods that delete frames, including Play Fair

Some methods remove a frame by **deleting** it, so the clip gets shorter: Shapley-drop, leave-one-out-drop and Play Fair. We ran them too:
- Shapley-drop: 64 orderings, 2 seeds, 120 videos;
- leave-one-out-drop: 120 videos;
- Play Fair (the authors' own code): 256 samples per subset size, 1 seed, the first 20 videos (27 targets).

### 6.1 Why these results aren't reliable for R3D

These methods show the model clips of 1 to 16 frames. But R3D's accuracy depends strongly on how many frames it gets, **even when the content is the same**. We tested this on 500 videos. We took k different frames and showed them in two ways: as a short clip of k frames, and with each frame repeated to fill 16 slots.

| k different frames | 1 | 2 | 4 | 6 | 8 | 9 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy as a short clip of k frames (%) | 15.6 | 17.4 | 25.6 | 35.0 | 36.4 | 64.8 | 66.8 | 71.8 | 74.4 | 79.4 |
| accuracy with the same frames repeated to 16 (%) | 63.8 | 72.8 | 78.4 | 77.6 | 77.6 | 79.0 | 79.8 | 79.8 | 79.2 | 79.4 |
| same prediction both ways (%) | 17.6 | 20.0 | 29.0 | 37.0 | 39.4 | 73.6 | 75.4 | 80.8 | 87.6 | 100 |

What this shows:
- **Short clips lose accuracy mostly because they're short, not because they contain less.** With 8 or fewer frames, the same frames are 41–52 percentage points more accurate when repeated to 16 slots.
- **The big jump between 8 and 9 frames comes from the network's design.** It appears for short clips (+28.4 points), but not for repeated clips (77.6% → 79.0%). At 8 frames or fewer, the network squeezes time down to a single position too early.
- **R3D mostly recognises videos from single frames.** One frame repeated 16 times is already correct for 63.8% of videos.

**Why this matters.** When a method deletes frames, adding a copy back changes two things: the clip's content (nothing new, since the frame is already there) and its length (one frame longer). For R3D, the length change alone can change the output a lot, especially around 8–9 frames. So part of each copy's credit comes from length, not content. Methods that fill in removed frames (freeze) don't have this problem: the clip always has 16 frames. **That's why our R3D conclusions use only the freeze-based and gradient-based methods.**

### 6.2 Play Fair gives the same values as Shapley-drop

Play Fair and Shapley-drop compute the same quantity. Both use the model's probability for the predicted class, with deleted frames and a uniform class prior for an empty clip. They only differ in how they pick random subsets of frames. So their results should differ only by random noise.

We tested this on the same 27 targets. As a measure of pure noise, we compared Shapley-drop with itself, using two different random seeds.

| | Play Fair vs Shapley-drop (seed 0) | Shapley-drop seed 1 vs seed 0 (noise only) |
|---|---|---|
| correlation of the per-frame scores (median) | 0.790 | 0.648 |
| relative difference ‖a − b‖ / ‖b‖ (median) | 0.252 | 0.372 |
| conditions where Play Fair is closer to seed 0 than seed 1 is | 91% | – |

Per target (exact copies; noisy copies within 0.01):

| measure | median difference, Play Fair − seed 0 [95% range] | median difference, seed 1 − seed 0 [95% range] | typical size of the difference: Play Fair vs seed 0 / seed 1 vs seed 0 |
|---|---|---|---|
| per-copy ratio, m = 4 | −0.015 [−0.113, +0.006] | −0.040 [−0.066, +0.052] | 0.050 / 0.084 |
| per-copy ratio, m = 6 | −0.045 [−0.095, +0.045] | −0.038 [−0.116, +0.032] | 0.095 / 0.117 |
| per-copy ratio, m = 8 | −0.046 [−0.130, +0.019] | −0.054 [−0.165, +0.074] | 0.066 / 0.140 |
| total ratio, m = 4 | −0.030 [−0.226, +0.013] | −0.080 [−0.131, +0.104] | 0.099 / 0.168 |
| total ratio, m = 6 | −0.134 [−0.285, +0.136] | −0.113 [−0.349, +0.096] | 0.285 / 0.351 |
| total ratio, m = 8 | −0.184 [−0.518, +0.075] | −0.215 [−0.661, +0.296] | 0.263 / 0.562 |
| β | −0.059 [−0.116, +0.028] | −0.067 [−0.126, +0.046] | 0.082 / 0.113 |

What this shows:
- **There's no systematic difference.** Every 95% range includes 0, and the differences are about as large as the noise between Shapley-drop's two seeds.
- **Play Fair is even closer to Shapley-drop than Shapley-drop is to itself.** That's expected, because this Play Fair run used more samples (about 2,650 model evaluations per clip, against about 1,000 for one Shapley-drop seed).
- **Comparing medians is misleading here.** On these 27 targets, the median β is −0.206 for Play Fair and −0.008 for Shapley-drop. That gap looks large. But per target, the two differ by only −0.059 in the middle case. With few targets and widely spread values, the median jumps around. So we compare each target with itself instead.

### 6.3 Results of the methods that delete frames

| method | β, target (exact) [95% range] | β, control | per-copy ratio, m = 4 / 6 / 8 | total ratio, m = 4 / 6 / 8 |
|---|---|---|---|---|
| Shapley-freeze (for comparison) | −0.37 [−0.40, −0.33] | −0.01 | 0.76 / 0.69 / 0.62 | 1.52 / 2.07 / 2.47 |
| Shapley-drop | −0.14 [−0.21, −0.09] | +0.02 | 0.89 / 0.87 / 0.84 | 1.79 / 2.60 / 3.38 |
| Play Fair (20 videos) | −0.21 [−0.27, −0.02] | +0.11 | 0.96 / 0.89 / 0.77 | 1.93 / 2.68 / 3.08 |
| Leave-one-out-freeze (for comparison) | −1.38 [−1.91, −1.07] | −0.11 | 0.05 / −0.02 / −0.05 | 0.10 / −0.05 / −0.19 |
| Leave-one-out-drop | −0.27 [−0.72, −0.15] | +0.17 | 0.33 / 0.16 / 0.01 | 0.65 / 0.49 / 0.02 |

Noisy copies give β within 0.07. The Play Fair row covers 20 videos and the Shapley-drop row 120, so their medians aren't directly comparable. Section 6.2 is the fair comparison.

What this shows:
- **Shapley-drop and Play Fair barely lower the credit per copy** (0.77–0.96). So the copies' total credit grows to about 3 times the reference at m = 8, while the model relies on the target only 1.28 times as much.
- **Leave-one-out-drop still collapses by m = 8** (0.01 per copy), but more slowly than the freeze version.
- **For R3D, these numbers mix content and clip length** (see 6.1). They should be checked on a model whose accuracy doesn't depend on clip length, such as V-JEPA2.

## 7. Conclusions

1. **The methods change a frame's credit when it's duplicated, even though the model doesn't change how much it relies on it.** The model keeps its prediction for most pairs (79–91% pass the gate). Its reliance on the target stays at 0.95–1.28 times the reference. Yet Shapley, Integrated Gradients and leave-one-out all lower the credit per copy. In the control clips, the target's credit doesn't change. So the change comes from duplicating the target.

2. **Each method gets it wrong in its own way.**
   - Leave-one-out: almost no credit per copy.
   - Integrated Gradients: the credit is split evenly, so each copy looks less important.
   - Shapley: less credit per copy, but too much in total.
   - Grad-CAM: too much in total.

3. **No method matches the model on both measures.** Either the credit per copy is too low, or the total is too high. So in videos with repeated or near-identical frames, a frame's credit depends on how often it appears, not only on how much the model uses it. Real videos often contain such repetition: static shots, slow motion, densely sampled clips.

4. **Noisy copies give the same results as exact copies** (β within 0.15). So the effect isn't caused by pixel-identical frames.

5. **The methods that delete frames, including Play Fair, can't be judged on R3D.** Play Fair and Shapley-drop agree with each other. But R3D's output depends strongly on clip length, so their results mix content with length. They need a model whose output doesn't depend on clip length.

## 8. Limitations

- **Selection.** Only videos the model gets right, and that have at least one important frame, are used: 44% of the videos screened. At higher m, fewer pairs pass the gate (149 at m = 8, against 172 at m = 4). The remaining pairs may favour videos that cope better with losing other frames. At m = 4 alone, the results show the same pattern: per-copy ratio 0.76 for Shapley, 0.54 for Integrated Gradients, 0.05 for leave-one-out.
- **The reference isn't the original clip.** It shows 8 frames, each twice. It still gives the same prediction as the original for 94% of videos.
- **The control isn't fully neutral at high m.** In the control clips, the model relies less on the target (0.42–0.80 times the reference). So read the results as the difference between target and control.
- **Sample size.** 120 videos were attributed, not the planned 300. The 95% ranges are narrow for Shapley and Integrated Gradients, and wider for leave-one-out.
- **Play Fair** was run on only 20 videos, with reduced settings.

## 9. Still to do

1. **Play Fair on all 120 videos** (optional for R3D, because of the clip-length problem). With 256 samples and 1 seed, this takes about 4 hours. With the full default settings (1,024 samples, 2 seeds), about 30 hours.
   ```
   python motivation/e1_attribute.py --model r3d --limit 120 --methods playfair --seeds 0 --pf-max-samples 256
   python motivation/e1_metrics.py attr --model r3d
   ```
2. **Figures** for these results.

The other models are covered in their own summaries (`R3D18_E1_summary.md`, `MC3_18_E1_summary.md`, `TRN_E1_summary.md`, `VJEPA2_E1_summary.md`, `VIDEOMAE_E1_summary.md`) and in `results/design_choice_report.md`. R3D-18 repeats these results. MC3-18 is the UCF101 model where the methods that delete frames, including Play Fair, can be judged without the clip-length problem.

## 10. Files and how to reproduce

Files in `results/e1/`:

| file | what it contains |
|---|---|
| `r3d_conditions.jsonl`, `r3d_skipped.jsonl` | the clips built for each video (m = 4, 6, 8), and the videos left out |
| `r3d_gate.csv` | the gate check (section 4) |
| `r3d_attributions.jsonl` | the per-frame scores for every clip, method and seed |
| `r3d_rows.csv`, `r3d_slopes.csv`, `r3d_summary.csv`, `r3d_beta.csv` | the measures: per row, per target, per method and m, and β with 95% ranges |
| `r3d_build.log`, `r3d_attribute.log` | run logs |
| `../frame_count_r3d_*.csv`, `../frame_count_r3d_short_*.csv` | accuracy against clip length, 1–24 frames |
| `../length_fixed_content_r3d_*.csv` | accuracy for k frames as a short clip vs repeated to 16 (section 6.1) |
| `r3d_insert_pilot/`, `r3d_realloc_pilot/`, `r3d_realloc500_m48/` | older runs, replaced by the current one |

Commands:
```
python motivation/e1_build_conditions.py --model r3d --limit 500 --ms 2 4 6 8
python motivation/e1_metrics.py gate --model r3d
python motivation/e1_attribute.py --model r3d --limit 120
python motivation/e1_attribute.py --model r3d --limit 120 --methods shapley_drop loo_drop
python motivation/e1_attribute.py --model r3d --limit 20 --methods playfair --seeds 0 --pf-max-samples 256
python motivation/e1_metrics.py attr --model r3d

# Play Fair vs Shapley-drop on the same clips (section 6.2)
python motivation/e1_compare_estimators.py --model r3d

# accuracy at 1-11 frames
python eval_frame_count.py --models r3d --frames 1 2 3 4 5 6 7 8 9 10 11 --limit 500 --datasets r3d=ucf101_test --out results/frame_count_r3d_short

# accuracy for k frames as a short clip vs repeated to 16 (section 6.1)
python eval_length_fixed_content.py --models r3d --datasets r3d=ucf101_test --ks 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --limit 500 --out results/length_fixed_content_r3d
```
