# Which video models can be used for E1, and how

E1 tests what attribution methods do when a frame appears several times in a clip. Not every video model is suitable for this test. This report explains which models we checked, what tests we ran on each, and what we decided.

We checked seven models:
- on **UCF101:** R3D-50, R3D-18, MC3-18 and VideoMAE;
- on **Something-Something v2 (SSv2):** V-JEPA2, TRN (official) and Play Fair's TRN.

**Result:**
- **Five models can be used:** R3D-50, R3D-18, MC3-18, V-JEPA2 and TRN (official).
- **VideoMAE can't be used.** Duplicating frames changes how it behaves, in a way that has nothing to do with the duplicated content.
- **Play Fair's TRN can't be tested.** Its files have been deleted from the internet.

## 1. What a model must satisfy

E1 says a method is wrong when a duplicated frame's credit changes for no good reason. For that to be fair, the model must meet four conditions:

1. **We run the model correctly.** With our frame sampling, preprocessing and class labels, it reaches its expected accuracy.
2. **Duplication changes nothing else.** The model must handle every clip we show it normally. This includes the duplicated clips, and the shorter or filled-in clips that attribution methods create when they remove frames. For example, if the model's accuracy depends on how many frames it gets, the test must avoid changing the number of frames.
3. **The model keeps its prediction** after duplication. We call this check the *gate*. Then we compare credit for the same predicted class.
4. **We know how much the model relies on the duplicated frame, and that reliance depends on the frame's content.** Duplication may change how much the model relies on a frame. That's fine, as long as we measure it and judge the methods against it. But if the model reacts the same way to any duplicated frame, whatever it shows, the test can't tell a wrong method from a correct one.

## 2. Three ways to duplicate a frame

| design | reference clip | target clip | control clip | clip length | how frames are removed |
|---|---|---|---|---|---|
| **insertion** | the original 16 frames, each once | m − 1 copies of the target t are **added** next to it | same length, but the extra frames are copies of other frames | 16 + m − 1 | deletion (`drop`) |
| **reallocation** | 8 frames (every other one), each shown **twice** | t gets m slots; to make room, some other frames drop to one copy | the same frames drop to one copy, but the freed slots go to the least important frame | stays fixed (16, or 8 for TRN) | filled in (`late` freeze) |
| **replacement** | the original 16 frames, each once | t's copies **replace** m − 1 neighbouring frames | the same frames are replaced with the least important frame | stays at 16 | deletion |

**How frames are removed.** Some attribution methods remove frames to see what changes. There are two ways:
- **deletion (`drop`):** the frame is taken out, and the clip gets shorter;
- **filling in (`late` freeze):** the gap is filled with the neighbouring frame (once from the frame before, once from the frame after, then the two results are averaged), so the length stays the same.

**Which design suits which model:**
- **Insertion** is the cleanest: nothing else in the clip changes. But the clip gets longer, so the model must not care about clip length.
- **Reallocation** keeps the length fixed. But every frame appears twice in the reference, so the model must not mind repeated frames.
- **Replacement** keeps both the length and the original clip as reference. But it removes neighbouring frames.

The design for each model is set in `MODEL_SPECS` in `motivation/e1_common.py`.

## 3. The tests we ran

| test | what it checks | how | condition |
|---|---|---|---|
| **T0. Checkpoint** | does the model reach its expected accuracy with our setup? | accuracy on the test set (`eval_accuracy.py` or a full test-set run) | 1 |
| **T1. Input limits** | which clip lengths can the model process at all? | run the model at other lengths; look at the architecture | 2 |
| **T2. Accuracy vs clip length** | does accuracy jump at some clip length? | `eval_frame_count.py`: sample T frames evenly across the whole video, for many T, on 500 test videos. Every extra frame is new content. So a smooth rise means "more information"; a sudden jump points to the architecture | 2 |
| **T3. Accuracy with the same frames** | does clip length matter when the content stays the same? | `eval_length_fixed_content.py`: take k different frames. Show them once as a short clip of k frames, and once with each frame repeated to fill 16 slots. 500 test videos | 2 (and whether repeated frames are a problem) |
| **T4. Reference check** | is the reference clip classified like the original clip? | compare predictions on the original clip and the doubled reference (reallocation only) | 2 |
| **T5. Gate** | does duplication keep the prediction? | for each m: how many target and control clips keep the reference prediction, and how many target–control pairs pass together (the *per-pair gate*) | 3 |
| **T6. Reliance on the target** | how much does the model rely on the duplicated frame? | the **I_t ratio**: the target's importance in the duplicated clip divided by its importance in the reference. Importance means how much the predicted class's probability drops when every copy of the frame is removed. Compared with the control clip | 4 |

T2 and T3 also decide whether the methods that **delete frames** can be used: Shapley-drop, leave-one-out-drop and Play Fair. These show the model clips of 1 to 24 frames. If clip length changes the model's output on its own, these methods mix up length and content.

**The per-pair gate.** A target clip and its control clip are used only if **both** keep the reference prediction. If either fails, both are left out. Every value of m is used, but only the pairs that pass. A stricter rule would drop a whole value of m when its overall agreement falls below 90%. We didn't use that rule.

## 4. Results for all models

| model | data | T0 accuracy | T1–T3 clip length | T4 reference | T5 pairs passing | T6 reliance on target (I_t ratio) | **decision** | methods that delete frames / Play Fair |
|---|---|---|---|---|---|---|---|---|
| **R3D-50** | UCF101 | 79.4% | jumps at 8/9 frames (+28.4 points) and 16/17 (−10.2); with the same frames, length 16 is 41–52 points more accurate | 94.1% same | 79–91% | 0.98–1.04 | **usable: reallocation** | not reliable (length effect) |
| **R3D-18** | UCF101 | 82.95% (reported 83.43%) | jumps at 4/5, 8/9, 16/17; up to +30 points with the same frames | 95.7% same | 92–95% | 1.02–1.33 | **usable: reallocation** | not reliable (length effect) |
| **MC3-18** | UCF101 | 86.83% (reported 87.05%) | flat from 3 to 24 frames; at most 0.4 points with the same frames | – (original clip is the reference) | 96–99% | **2.01–3.19** (grows with m) | **usable: insertion** | **usable** (small confidence effect) |
| **V-JEPA2** | SSv2 | 68.2% | smooth; flat from 10 to 24 frames; at most 3.8 points with the same frames | – (original clip is the reference) | 89–93% | 0.85–0.95 | **usable: insertion** | **usable** |
| **TRN (official)** | SSv2 | about 30–34% | takes exactly 8 frames | 87.8% same | 54% (only m = 4) | 1.00 | **usable with limits: reallocation** | not possible (fixed 8 frames) |
| **VideoMAE** | UCF101 | 76.0% | at most 16 frames; repeated frames in a pair cost up to 20.8 points | **37.2% same** (reallocation) | 80–96% (replacement) | **negative** for any duplicated frame | **not usable** | – |
| **TRN (Play Fair)** | SSv2 | – | – | – | – | – | **not available** (files deleted) | – |

The T0 accuracies for R3D-50, V-JEPA2 and VideoMAE are on the 500 test videos used for T2–T3. For MC3-18 and R3D-18 they're on the full test set. SSv2 accuracies are on `ssv2_sampled`.

![Accuracy change with clip length](figures/fig1_accuracy_vs_length.png)

**Figure 1.** T2 for R3D-50, V-JEPA2 and VideoMAE. Each line shows accuracy at T frames, minus the model's own accuracy at 16 frames. 500 test videos per model, frames sampled evenly across the whole video (no duplicates). The grey bands mark R3D-50's two problem ranges (8 frames or fewer, and 17 or more). The models use different datasets, so compare the changes, not the accuracy levels.

![Accuracy at fixed content](figures/fig3_fixed_content.png)

**Figure 3.** T3 for R3D-50, V-JEPA2 and VideoMAE. For each k, the same k frames are shown as a short clip of k frames, and repeated to fill 16 slots. A gap between the two lines means clip length matters even with the same content:
- **R3D-50** is much more accurate with 16 slots;
- **V-JEPA2** shows no gap;
- **VideoMAE** is *less* accurate when frames are repeated. The loss grows with the number of frame pairs that hold two identical frames.

---

## 5. R3D-50 (UCF101) → usable, with reallocation

**The model.** A 3D ResNet-50 from the 3D-ResNets-PyTorch code (Hara et al.). Pretrained on Kinetics, then fine-tuned on UCF101 (`models/r3d/ucf101.json`, `models/r3d/ckpt/save_200.pth`). It takes 16 frames of 112 × 112 pixels. Its first layers include a pooling step that halves the number of time steps, and three later stages halve it again.

**First try: insertion. It failed.** R3D-50 accepts clips of any length, so we first used insertion. In a 20-video trial, predictions changed in up to 29% of cases, for the target **and** for the control. The probability of the predicted class fell by about 0.46 in **every** case. The control duplicates other frames, so the cause had to be the longer clip, not the duplicated frame.

**Why clip length matters for R3D-50.** Each of its four time-halving steps turns T time steps into ⌊(T − 1)/2⌋ + 1:

| T | after pooling | layer2 | layer3 | layer4 (into the final average) | what this means |
|---|---|---|---|---|---|
| 8 or fewer | 4 or fewer | 2 or fewer | **1** | 1 | layer4 sees only a single time step |
| 9–16 | 5–8 | 3–4 | **2** | **1** | **the case the model was trained on** |
| 17 | 9 | 5 | 3 | **2** | two time steps reach the final average, mostly made of padding at the edges |
| 24 | 12 | 6 | 3 | 2 | same |

**T2 (Table 1, Figure 1).** Accuracy jumps twice, once at each boundary:
- from 16 to 17 frames it **drops 10.2 points** (74 videos become wrong, 23 become right);
- from 8 to 9 frames it **rises 28.4 points**.

Between the boundaries, it rises gradually.

**Table 1.** R3D-50 on 500 UCF101 test videos, frames sampled evenly. "Correct → wrong" counts videos compared with 16 frames.

| T | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | **9** | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy (%) | 15.6 | 17.4 | 24.8 | 25.6 | 31.8 | 35.0 | 37.0 | **36.4** | **64.8** | 66.8 | 70.8 | 71.8 |
| change vs 16 frames (points) | −63.8 | −62.0 | −54.6 | −53.8 | −47.6 | −44.4 | −42.4 | **−43.0** | **−14.6** | −12.6 | −8.6 | −7.6 |
| correct → wrong | 326 | 314 | 282 | 276 | 249 | 233 | 222 | 226 | 90 | 77 | 61 | 55 |
| wrong → correct | 7 | 4 | 9 | 7 | 11 | 11 | 10 | 11 | 17 | 14 | 18 | 17 |

| T | 13 | 14 | 15 | **16** | **17** | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy (%) | 75.4 | 74.4 | 77.0 | **79.4** | **69.2** | 68.4 | 68.2 | 69.6 | 68.8 | 68.6 | 67.0 | 66.4 |
| change vs 16 frames (points) | −4.0 | −5.0 | −2.4 | 0 | **−10.2** | −11.0 | −11.2 | −9.8 | −10.6 | −10.8 | −12.4 | −13.0 |
| correct → wrong | 33 | 33 | 22 | – | 74 | 76 | 77 | 73 | 79 | 80 | 88 | 91 |
| wrong → correct | 13 | 8 | 10 | – | 23 | 21 | 21 | 24 | 26 | 26 | 26 | 26 |

**T3 (Table 3b, Figure 3).** With 8 or fewer frames, the same frames are correct for 16–36% of videos as a short clip, but 64–80% when repeated to 16 slots. So R3D-50 loses accuracy on short clips mainly because they're short, not because they contain less. It also recognises videos mostly from single frames: one frame repeated 16 times is correct for 63.8% of videos.

**Decision: reallocation, with removed frames filled in (m = 2 as reference, 4, 6, 8).** Every clip, including every clip the attribution methods create, then has exactly 16 frames. That's inside the 9–16-frame range where R3D-50 works normally. The methods that delete frames still mix up length and content (see `results/e1/R3D_E1_summary.md`).

**T4–T6 (Figure 2, Table 2).** We ran the gate on 500 test videos. 220 videos were kept (102 were classified wrongly, 178 had no frame that mattered enough), giving 343 targets.

**Table 2.** R3D-50 gate, exact copies (noisy copies within about 1 point).

| design | clip | m | targets | same prediction (%) | change in predicted-class probability | I_t ratio |
|---|---|---|---|---|---|---|
| insertion (trial) | target | 2 / 4 / 8 | 17 | 82.4 / 88.2 / 70.6 | −0.47 / −0.44 / −0.50 | −17.4 / −17.1 / −17.3 |
| insertion (trial) | control | 2 / 4 / 8 | 17 | 88.2 / 100 / 82.4 | −0.46 / −0.46 / −0.49 | −16.1 / −0.05 / −0.10 |
| reallocation | original vs doubled reference | – | 220 | 94.1 | −0.001 | – |
| reallocation | target | 4 / 6 / 8 | 343 | 93.6 / 87.5 / 85.4 | −0.020 / −0.067 / −0.110 | 0.98 / 1.01 / 1.04 |
| reallocation | control | 4 / 6 / 8 | 343 | 93.3 / 87.2 / 81.9 | −0.028 / −0.077 / −0.127 | 0.80 / 0.57 / 0.42 |

What this shows:
- **With insertion, the I_t ratio was about −17.** Removing the copies made the clip shorter than 17 frames again, and that *raised* the probability. So this number measured clip length, not the target.
- **With reallocation, the model relies on the target as much as before** (I_t ratio 0.98–1.04).
- **The prediction changes more often at higher m, but the control changes just as often.** So the cause isn't the duplicated target. It's the other frames that lose a copy (2, 4 or 6 of them).
- **Showing every frame twice changes the prediction in only 5.9% of videos,** and the confidence hardly moves (−0.001).

**Which pairs we used.** If we required 90% agreement for a whole value of m, only m = 4 would pass. Instead we use the **per-pair gate** (section 3). It keeps 91.0%, 82.2% and 79.0% of pairs at m = 4, 6 and 8. The pairs that remain may favour videos that cope well with losing other frames.

![R3D sensitivity gate](figures/fig2_r3d_gate.png)

**Figure 2.** R3D-50 gate with exact copies. Left: the insertion trial (20 videos, 17 targets). Right: reallocation (500 videos, 343 targets). Top: share of clips that keep the reference prediction. Bottom: average change in the predicted class's probability.

**Decision: usable, with reallocation and the methods that fill in removed frames.** E1 results: `results/e1/R3D_E1_summary.md` (120 videos).

---

## 6. R3D-18 (UCF101) → usable, with reallocation

**The model.** torchvision's `r3d_18`, pretrained on Kinetics-400 and fine-tuned on UCF101 split 1. It was published as `dronefreak/r3d-18-ucf101` on the Hugging Face Hub (our wrapper: `models/torchvision_ucf101.py`). It takes 16 frames of 112 × 112 pixels. Three stages halve the number of time steps. Unlike R3D-50, it has no extra pooling step at the start.

**T0.** We copied the publisher's setup: 16 frames sampled evenly with `linspace`, resized to 128 × 171, cropped to 112, and normalised with the Kinetics mean and standard deviation. One detail mattered: the class labels. The model numbers its classes in alphabetical order of the folder names (Python's `sorted()`). The publisher's own prediction code uses the order of `classInd.txt` instead. The two orders differ for four classes: HammerThrow/Hammering and JumpRope/JumpingJack.
- **With the alphabetical order:** 82.95% on the 3,782 readable test videos (reported: 83.43%).
- **With the `classInd.txt` order:** 80.41%, and 0% on those four classes.

So the alphabetical order is correct.

**T2–T3 (Figure 4, Tables 5 and 5b).** Accuracy jumps at 4 → 5 frames (+12.8 points), 8 → 9 (+6.2) and 16 → 17 (−4.6). These are the lengths where the number of time steps after its three halving stages changes. It's the same cause as in R3D-50, at shifted lengths. With the same frames, 16 slots are up to 30 points more accurate (67.8% against 37.8% for 1 frame). Repeated frames cause no problem, because the model reads frames one at a time.

**T4–T6.** We ran the gate on 500 test videos. 345 videos were kept (95 classified wrongly, 60 with no frame that mattered enough), giving 547 targets (345 high, 202 mid).

| check | m = 4 | m = 6 | m = 8 |
|---|---|---|---|
| original clip vs doubled reference | 95.7% same, probability change +0.004 | | |
| target clip: same prediction / probability change | 96.7% / +0.002 | 96.2% / +0.008 | 94.9% / −0.002 |
| control clip: same prediction / probability change | 96.7% / −0.006 | 96.9% / −0.002 | 94.5% / −0.009 |
| I_t ratio, target / control | 1.02 / 0.95 | 1.17 / 0.91 | 1.33 / 0.86 |
| pairs passing the per-pair gate | 95.2% | 94.9% | 91.8% |

**Decision: usable, with reallocation and the methods that fill in removed frames (m = 2, 4, 6, 8).** The reference is classified like the original clip. Duplication keeps the prediction for about 95% of targets. The model relies on the target about as much as before. R3D-18 is a check that the R3D-50 results hold in a different implementation. The attribution run is still to do.

---

## 7. MC3-18 (UCF101) → usable, with insertion

**The model.** torchvision's `mc3_18`, pretrained on Kinetics-400 and fine-tuned on UCF101 split 1 (`dronefreak/mc3-18-ucf101`). The first stage looks across frames; the later stages look at each frame on its own. The number of time steps is never reduced. At the end, the model averages over all time steps.

**T0.** Same setup and same class-order finding as R3D-18. **86.83%** on the 3,782 readable test videos with the alphabetical order (reported: 87.05%), against 84.21% with the `classInd.txt` order.

**T2–T3 (Figure 4, Tables 5 and 5b).**
- **Accuracy doesn't jump at any length.** Clips of 17 to 24 frames are as accurate as 16 frames (87.6–88.6%).
- **With the same frames, clip length doesn't change accuracy** (at most 0.4 points apart; same prediction for 87–99% of videos).
- **Clip length has a small effect on confidence.** With few different frames, the short clip gives a *higher* probability for the true class than the 16-slot clip: 0.74 against 0.57 for 1 frame, 0.67 against 0.64 for 8 frames, and equal at 14.
- **The model recognises videos mostly from single frames** (78.2% from one frame). So many videos have no single frame that matters enough to be a target.

**Table 5.** MC3-18 and R3D-18: accuracy (%) for T frames, sampled evenly, 500 test videos.

| T | 1 | 2 | 4 | 5 | 8 | 9 | 12 | **16** | 17 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MC3-18 | 78.2 | 85.4 | 86.8 | 87.0 | 87.2 | 87.6 | 87.6 | **88.0** | 87.8 | 88.6 | 87.6 | 88.2 |
| R3D-18 | 37.8 | 52.0 | 63.0 | 75.8 | 74.6 | 80.8 | 81.6 | **84.4** | 79.8 | 81.8 | 81.0 | 82.0 |

**Table 5b.** Accuracy (%) for the same k frames, as a short clip and repeated to 16 slots.

| k | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|
| MC3-18, short clip | 78.2 | 85.4 | 86.8 | 87.0 | 87.2 | 87.2 | 87.6 | 87.8 |
| MC3-18, repeated to 16 | 78.6 | 85.0 | 86.4 | 87.2 | 87.2 | 87.0 | 87.2 | 87.8 |
| R3D-18, short clip | 37.8 | 52.0 | 63.0 | 76.2 | 74.6 | 82.4 | 81.6 | 83.6 |
| R3D-18, repeated to 16 | 67.8 | 75.4 | 81.4 | 82.4 | 81.8 | 82.6 | 83.4 | 84.6 |

**T5–T6.** We ran the gate on 500 test videos. 265 videos were kept (71 classified wrongly, 164 with no frame that mattered enough), giving 417 targets (265 high, 152 mid).

| check | m = 3 | m = 5 | m = 9 |
|---|---|---|---|
| target clip: same prediction / probability change | 99.5% / **+0.024** | 99.5% / **+0.037** | 98.8% / **+0.051** |
| control clip: same prediction / probability change | 98.8% / −0.009 | 98.8% / −0.016 | 96.9% / −0.029 |
| I_t ratio, target / control | **2.01** / 0.90 | **2.55** / 0.83 | **3.19** / 0.70 |
| pairs passing the per-pair gate | 98.6% | 98.3% | 95.9% |

**The model relies more on a frame when it's duplicated.** Removing all copies of the target lowers the probability 2.0 to 3.2 times as much as removing the single original frame. Duplicating the target also raises the probability of the predicted class. This follows from the design. The model averages over all time steps, and the m copies fill m of those steps, so the target counts more in the average.

**This reaction depends on the content.** When the least important frame is duplicated instead (the control), the model relies *less* on the target (0.90 → 0.70), because the target is now a smaller part of the clip. VideoMAE is different (section 9): it reacted the same way to any duplicated frame.

**What this means for judging the methods.** For MC3-18, a correct method should *not* leave the target's credit unchanged. It should follow the model. In total, the copies should get about 2.0, 2.6 and 3.2 times the reference credit at m = 3, 5 and 9. Per copy, that's about 0.67, 0.51 and 0.35 (the I_t ratio divided by m). For comparison, an even split of a fixed total would give 0.33, 0.20 and 0.11 per copy.

**Decision: usable, with insertion and the methods that delete frames, including Play Fair (m = 1, 3, 5, 9).** MC3-18 is the UCF101 model where Play Fair can be tested without R3D's clip-length problem. The methods must be judged against the model's higher reliance on the target. The attribution run is still to do (see section 12 for a speed problem).

![torchvision UCF101 models](figures/fig4_torchvision_length.png)

**Figure 4.** T2 and T3 for MC3-18 and R3D-18, 500 UCF101 test videos. Left: accuracy change compared with 16 frames, for different clip lengths. Middle and right: accuracy for the same frames as a short clip and repeated to 16 slots.

---

## 8. V-JEPA2 (SSv2) → usable, with insertion

**The model.** `facebook/vjepa2-vitl-fpc16-256-ssv2`. It works out each token's position from its index, so it has no fixed-size position table. It processes frames in pairs. It doesn't reduce the number of time steps, and its final layer can take any number of tokens.

**T2 (Table 3, Figure 1).** Accuracy rises smoothly and then levels off. Each extra frame adds 13.8 points at first (1 → 2 frames), but only about 1 point later (8 → 10). From 10 to 24 frames it's flat: at 18, 20 and 24 frames it's within 0.4 points of 16 frames. Videos that change their prediction go both ways about equally.

**Table 3.** V-JEPA2 on 500 `ssv2_sampled` videos, frames sampled evenly.

| T | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 15 | **16** | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| used in E1 as | | | | | | | | | | reference (m = 1) | m = 3 | m = 5 | m = 9 |
| accuracy (%) | 12.2 | 26.0 | 47.2 | 57.0 | 64.6 | 66.8 | 67.6 | 69.0 | 67.2 | **68.2** | 68.4 | 68.2 | 67.8 |
| change vs 16 frames (points) | −56.0 | −42.2 | −21.0 | −11.2 | −3.6 | −1.4 | −0.6 | +0.8 | −1.0 | 0 | +0.2 | 0 | −0.4 |

**T3 (Table 3b, Figure 3).**

**Table 3b.** Accuracy for the same k frames, as a short clip and repeated to 16 slots, 500 videos (R3D-50 shown for comparison).

| k different frames | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| V-JEPA2, short clip (%) | 12.2 | 26.0 | 47.2 | 57.0 | 64.6 | 66.8 | 67.6 | 69.0 | 68.2 |
| V-JEPA2, repeated to 16 (%) | 14.4 | 24.4 | 50.4 | 60.8 | 62.4 | 66.2 | 68.6 | 67.8 | 68.2 |
| R3D-50, short clip (%) | 15.6 | 17.4 | 25.6 | 35.0 | 36.4 | 66.8 | 71.8 | 74.4 | 79.4 |
| R3D-50, repeated to 16 (%) | 63.8 | 72.8 | 78.4 | 77.6 | 77.6 | 79.8 | 79.8 | 79.2 | 79.4 |

What this shows:
- **For V-JEPA2, the two ways differ by at most 3.8 points, in both directions.** So clip length doesn't matter when the content is the same.
- **Accuracy depends on how many different frames there are.** One frame repeated 16 times gives only 14.4%. That makes sense, because SSv2 actions depend on movement over time.
- **Repeated frames in a pair cause no problem** (62.4% against 64.6% at k = 8, where every pair holds two identical frames).

**Design: insertion, with odd m (m = 1, 3, 5, 9).** With odd m, the added copies fill whole frame pairs, so all the original pairs stay together. With even m, every later pair would shift by one frame.

**T5–T6.** We ran the gate on 72 videos. 34 were kept (26 classified wrongly, 12 with no frame that mattered enough), giving 57 targets. The target clip keeps the prediction in 94.7%, 96.5% and 94.7% of cases at m = 3, 5 and 9 (control: 96.5%, 91.2%, 89.5%). The I_t ratio is 0.95, 0.94 and 0.85. Between 89.5% and 93.0% of pairs pass the per-pair gate.

**Notes.**
- At m = 3, two copies share one frame pair, and the original sits in a pair with its neighbour. So the number of pairs that contain the target is 1, 2, 3 and 5 for m = 1, 3, 5 and 9.
- When a frame is deleted, all later frames shift into new pairs. T3 doesn't test this shift directly.

**Decision: usable, with insertion and the methods that delete frames, including Play Fair.** V-JEPA2 is the SSv2 model where Play Fair can be tested. E1 results: `results/e1/VJEPA2_E1_summary.md` (34 videos). The official Play Fair code hasn't been run on it yet, but Shapley-drop computes the same quantity.

---

## 9. VideoMAE (UCF101) → not usable

**The model.** `nateraw/videomae-base-finetuned-ucf101`. It takes 16 frames of 224 × 224 pixels, processes frames in pairs, and has a **fixed** position table with room for exactly 16 ÷ 2 × (224/16)² = **1,568 tokens**.

**T0.** With the installed `transformers` library (version 5.8.0.dev0), part of the model's weights (the q/v attention biases) didn't load and were left at zero. `models/videomae_ucf101.py` now fixes this. Accuracy is 76.0% on the 500 test videos.

**T1 (Table 4).** A clip longer than 16 frames has more tokens than the position table has rows, so the model stops with an error. Insertion is impossible.

**Table 4.** Running the model at other lengths.

| T | tokens | result |
|---|---|---|
| 16 | 1568 | runs |
| 18 | 1764 | `RuntimeError: size of tensor a (1764) must match ... (1568)` |
| 24 | 2352 | `RuntimeError: size of tensor a (2352) must match ... (1568)` |

**T2–T3 (Figures 1 and 3, Table 4b).** Shorter clips work, because we use only the first part of the position table. Accuracy rises smoothly from 42.4% (2 frames) to 76.0% (16 frames). With the same frames, the short clip is never less accurate than the repeated one. So shorter clips aren't a problem.

But repeated frames are. VideoMAE processes two neighbouring frames together. When both frames of a pair are the same, accuracy drops:
- when **every** pair holds two identical frames (k = 4 and k = 8), accuracy falls by 13.2 and **20.8** points;
- when **half or fewer** do (k = 10, 12, 14), it falls by only 0.4 to 6.8 points.

**Table 4b.** VideoMAE, accuracy for the same k frames (500 videos).

| k different frames | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| short clip (%) | 42.4 | 56.2 | 63.2 | 66.4 | 68.4 | 73.6 | 74.8 | 76.0 |
| repeated to 16 (%) | 40.4 | 43.0 | 53.4 | 45.6 | 67.2 | 66.8 | 74.4 | 76.0 |
| pairs holding two identical frames (repeated clip) | 8 / 8 | 8 / 8 | 6 / 8 | 8 / 8 | 4 / 8 | 4 / 8 | 2 / 8 | 0 / 8 |

**T4: reallocation failed.** In the reallocation reference, every frame appears twice in a row, so every pair holds two identical frames. On 50 videos (`results/e1/videomae_diag`; 43 kept), the reference gives the same prediction as the original clip in only **37.2%** of videos. For comparison: R3D-50 94.1%, R3D-18 95.7%, TRN 87.8%. The probability of the original prediction falls by 0.36 on average, and accuracy falls from 79% to 37%. So the reference doesn't show how the model behaves on real clips.

**T5–T6: replacement also failed.** We built the replacement design for VideoMAE (`design="replace"`). The reference is the original 16 frames. The target's copies replace m − 1 neighbouring frames, in whole pairs. On 50 videos (`results/e1/videomae_replace_diag`; 33 kept, 57 targets), the prediction is kept: 96.4%, 96.2%, 85.7% and 86.7% of target clips at m = 2, 4, 6 and 8, and 80–96% of pairs pass. But the model's reliance on **any** duplicated frame becomes **negative**:

| m | target's importance, reference | target's importance, duplicated | least important frame, reference | least important frame, duplicated (control) |
|---|---|---|---|---|
| 2 | +0.052 | −0.011 | +0.001 | −0.020 |
| 4 | +0.063 | −0.028 | +0.001 | −0.033 |
| 6 | +0.065 | −0.058 | +0.001 | −0.049 |
| 8 | +0.066 | −0.062 | +0.001 | −0.083 |

Removing the copies *raises* the probability of the predicted class. This happens for the important target and for the unimportant frame alike, and it gets stronger as m grows. Once duplicated, the target's importance can't be told apart from the unimportant frame's, even though it was about 50–65 times larger in the reference. This isn't caused by how we delete frames: in the reference, a frame's importance doesn't depend on its position (correlation −0.025).

**Decision: not usable.** VideoMAE reacts to pairs of identical frames, and any duplication creates them, whatever the frame shows. So its reliance on a duplicated frame doesn't depend on the frame's content (condition 4 fails). A method that gives the copies less credit could simply be right. One alternative would be to duplicate whole frame pairs instead of single frames, so no pair holds two identical frames. That changes what is duplicated, and we didn't pursue it. Details: `results/e1/VIDEOMAE_E1_summary.md`.

---

## 10. TRN, official version (SSv2) → usable with limits, with reallocation

**The model.** The original TRN-pytorch model, trained on SSv2 (`TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar`). It turns each of exactly **8** frames into a feature vector of 256 numbers. Then it combines the features of different frame subsets with "relation heads". Each head expects a fixed input size:

| frames combined | 8 | 7 | 6 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|---|---|
| input size | 2048 | 1792 | 1536 | 1280 | 1024 | 768 | 512 |
| possible frame subsets | 1 | 8 | 28 | 56 | 70 | 56 | 28 |
| subsets used per run | 1 | 3 | 3 | 3 | 3 | 3 | 3 |

The model picks its frame subsets at random each time it runs. Our code fixes this randomness, so the same clip always gives the same output.

**T0.** The file only loads with `torch.load(weights_only=False)`, because newer PyTorch versions block it by default. Accuracy is about 30% on `ssv2_sampled` (30.0% on 60 videos; 33.6% on the full SSv2 test folder).

**T1.** The model needs exactly 8 frames, so only reallocation over 8 slots is possible. That means 4 contents, with m = 2 as reference and **m = 4** as the only other level (m can't exceed the number of contents).

The methods that delete frames can't be used, and neither can Play Fair. Play Fair handles fixed-length models like TRN in a special way. It trains a separate TRN for each number of frames (1, 2, …, 8) and combines them into a "multi-scale" model. For a set of frames, this model averages the outputs of all the single-size models over all smaller subsets of those frames. Play Fair then explains that combined model. These separately trained models aren't available (section 11). The relation heads of the official TRN are a different thing: they were trained together inside one 8-frame model. Using them alone would create a different model.

**T4–T6.** We ran the condition build on all 1,044 `ssv2_sampled` videos. 245 were kept (778 classified wrongly, 21 with no frame that mattered enough), giving 375 targets (245 high, 130 mid).

| check (m = 4) | same prediction | probability change | I_t ratio |
|---|---|---|---|
| original 8 frames vs doubled reference | **87.8%** | −0.047 | – |
| target clip | **72.3%** | −0.142 | 1.00 (all targets); 1.32 (pairs that passed) |
| control clip | **60.0%** | −0.241 | 0.58 (all); 0.98 (pairs that passed) |

**202 of 375 pairs (53.9%)** pass the per-pair gate, from 166 videos. "Mid" targets pass less often (34%) than "high" targets (64%).

**Decision: usable, but with limits. Reallocation, with the methods that fill in removed frames.**
- **Why it's usable:** the reference is classified like the original clip for 88% of videos. In the pairs that pass, the model relies on the target as much or more after duplication (I_t ratio 1.32). This reaction depends on the content: in the control, the model's reliance on the target stays at 0.98.
- **The limits:**
  - **only one level of duplication** (m = 4), so there's a single ratio instead of a trend over m;
  - **a big change to the clip:** at m = 4, two of the four contents drop to one copy, so only 54% of pairs pass and the remaining pairs are a narrow selection;
  - **low accuracy,** so most videos are left out;
  - **no methods that delete frames.**

E1 results: `results/e1/TRN_E1_summary.md`.

## 11. TRN, Play Fair's version → not available

Play Fair's own TRN files were stored on the authors' Dropbox. These include the backbone (`trn.pth`), the 8-frame relation head (`trn_8_frames.pth`), the single-size models (`trn_1_frames.pth` … `trn_16_frames.pth`) and the combined multi-scale models (`mtrn_8_frames.pth`, `mtrn_16_frames.pth`). Every link now says "File Deleted", including the links in Play Fair's own `download.sh`. Our local copies are just these "File Deleted" web pages. We can only test this model if we get the files from the authors or from someone who kept a copy.

---

## 12. Still to do

1. **MC3-18 and R3D-18 attribution.** The clips and gates are ready.
   - **Speed problem with MC3-18:** when 16 clips of 18 frames or more are processed together in full precision, one batch takes 1.3–9 seconds, against 0.18 seconds at 17 frames. That makes Shapley-drop at m = 9 about 20 times slower than expected. Smaller batches don't have this problem. We need to choose a fix (for example, a limit on frames per batch) before the full run.
2. **V-JEPA2:** run the official Play Fair code, to compare it directly with Shapley-drop. Use more videos if there's time.
3. **VideoMAE:** not usable. Optional: try duplicating whole frame pairs instead of single frames.
4. **Play Fair's TRN:** not available.

## 13. How to reproduce

```
# T0: accuracy (torchvision models use the alphabetical class order by default)
python eval_accuracy.py --model mc3_18 --dataset ucf101_test
python eval_accuracy.py --model r3d_18 --dataset ucf101_test

# T2 (Figure 1, Tables 1, 3, 5)
python eval_frame_count.py --models r3d --frames 12 13 14 15 16 17 18 19 20 21 22 23 24 --limit 500 --datasets r3d=ucf101_test --out results/frame_count_r3d
python eval_frame_count.py --models r3d --frames 1 2 3 4 5 6 7 8 9 10 11 --limit 500 --datasets r3d=ucf101_test --out results/frame_count_r3d_short
python eval_frame_count.py --models vjepa2 --frames 16 18 20 24 --limit 500 --datasets vjepa2=ssv2_sampled
python eval_frame_count.py --models vjepa2 --frames 1 2 4 6 8 10 12 14 15 16 --limit 500 --datasets vjepa2=ssv2_sampled --out results/frame_count_vjepa2_short
python eval_frame_count.py --models videomae --frames 2 4 6 8 10 12 14 16 --limit 500 --datasets videomae=ucf101_test --out results/frame_count_videomae
python eval_frame_count.py --models mc3_18 r3d_18 --frames 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 24 --limit 500 --datasets mc3_18=ucf101_test r3d_18=ucf101_test --out results/frame_count_tv

# T3 (Figure 3, Tables 3b, 4b, 5b)
python eval_length_fixed_content.py --models r3d --datasets r3d=ucf101_test --ks 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --limit 500 --out results/length_fixed_content_r3d
python eval_length_fixed_content.py --models vjepa2 --datasets vjepa2=ssv2_sampled --ks 1 2 4 6 8 10 12 14 16 --limit 500 --out results/length_fixed_content_vjepa2
python eval_length_fixed_content.py --models videomae --datasets videomae=ucf101_test --ks 2 4 6 8 10 12 14 16 --limit 500 --out results/length_fixed_content_videomae
python eval_length_fixed_content.py --models mc3_18 r3d_18 --datasets mc3_18=ucf101_test r3d_18=ucf101_test --ks 1 2 4 6 8 10 12 14 16 --limit 500 --out results/length_fixed_content_tv

# T4-T6: build the clips and run the gate (current MODEL_SPECS)
python motivation/e1_build_conditions.py --model r3d --limit 500 --ms 2 4 6 8
python motivation/e1_build_conditions.py --model r3d_18 --limit 500
python motivation/e1_build_conditions.py --model mc3_18 --limit 500
python motivation/e1_build_conditions.py --model vjepa2 --limit 72 --fp16          # run in Colab
python motivation/e1_build_conditions.py --model trn_official
python motivation/e1_metrics.py gate --model <model>

# R3D-50 insertion trial (Figure 2, left; needs design="insert" for r3d in MODEL_SPECS)
python motivation/e1_build_conditions.py --model r3d --limit 20

# VideoMAE design checks
# reallocation (needs design="realloc", removal="late" for videomae):
python motivation/e1_build_conditions.py --model videomae --limit 50 --ms 2 --dup-types exact --include-wrong --out results/e1/videomae_diag
# replacement (current MODEL_SPECS):
python motivation/e1_build_conditions.py --model videomae --limit 50 --fp16 --out results/e1/videomae_replace_diag

# figures (needs only matplotlib; in tc_env, numpy crashes inside matplotlib, so use another environment)
python results/design_choice_figures.py
```

| file | what it contains |
|---|---|
| `results/frame_count_r3d*_*.csv`, `results/frame_count_summary.csv`, `results/frame_count_vjepa2_short_*.csv`, `results/frame_count_videomae_*.csv`, `results/frame_count_tv_*.csv` | T2: accuracy for different clip lengths (summary and per-video predictions) |
| `results/length_fixed_content_r3d_*.csv`, `results/e1/length_fixed_content_vjepa2_*.csv`, `results/length_fixed_content_videomae_*.csv`, `results/length_fixed_content_tv_*.csv` | T3: accuracy for the same frames as a short clip and repeated to 16 |
| `results/tv_ucf101_accuracy_preds.csv`, `results/tv_ucf101_accuracy.log` | T0 for MC3-18 and R3D-18 on the full test set, with both class orders |
| `results/e1/<model>_conditions.jsonl`, `<model>_skipped.jsonl`, `<model>_gate.csv` | T4–T6 for r3d, r3d_18, mc3_18, vjepa2 and trn_official |
| `results/e1/r3d_insert_pilot/` | R3D-50 insertion trial |
| `results/e1/videomae_diag/`, `results/e1/videomae_replace_diag/` | VideoMAE design checks |
| `results/figures/fig1_accuracy_vs_length.png`, `fig2_r3d_gate.png`, `fig3_fixed_content.png`, `fig4_torchvision_length.png` | figures |
| `results/e1/R3D_E1_summary.md`, `TRN_E1_summary.md`, `VJEPA2_E1_summary.md`, `VIDEOMAE_E1_summary.md` | E1 results for each model |
