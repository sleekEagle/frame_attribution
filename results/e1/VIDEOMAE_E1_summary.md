# E1 summary: VideoMAE on UCF101 — not usable for E1

## In short

We **don't use VideoMAE** for the E1 test.

E1 needs a model that relies on a frame about as much after it's duplicated as before. Then, if a method's credit for the frame changes, we know the method is at fault.

VideoMAE doesn't meet this. It reads frames in pairs, and when both frames of a pair are the same, its output changes, whatever the frames show. So duplicating a frame changes the model itself. We tried both duplication designs that VideoMAE's input allows, and both fail:
- **Reallocation:** the reference clip (each frame twice) already changes the prediction in 63% of videos.
- **Replacement:** removing the copies of *any* frame raises the predicted class's probability, even for the frame the model relied on most. So the model's own measure of importance stops making sense.

This document records the evidence.

## 1. The model

**Model.** `nateraw/videomae-base-finetuned-ucf101`: a ViT-B that takes 16 frames at 224 px.
- It reads frames in **pairs** (tubelets of 2): each pair of neighbouring frames becomes one set of tokens.
- Its position table has room for exactly 1,568 tokens, which is 16 frames.
- The checkpoint's attention biases (q/v) aren't loaded by `transformers` 5.8.0.dev0, so our wrapper (`models/videomae_ucf101.py`) adds them back.

**Data.** The UCF101 split-1 test set.

## 2. What inputs VideoMAE accepts

| question | what we found | what it means |
|---|---|---|
| clips longer than 16 frames? | an error at 18 and 24 frames: more tokens than the position table has room for | we can't add copies (insertion is impossible) |
| clips shorter than 16 frames? | they run, using the first part of the position table. Accuracy rises smoothly from 42.4% (2 frames) to 76.0% (16 frames), with no sudden jumps (500 videos) | deleting frames is possible |
| does length matter, with the same content? | a short clip of k frames is at least as accurate as the same k frames repeated to 16, for every k (500 videos) | no length problem like R3D's |
| pairs of two identical frames? | accuracy falls with the number of such pairs: −13.2 points at k = 4 and −20.8 at k = 8, where every pair has two identical frames; −0.4 to −6.8 points where half the pairs or fewer do | **putting copies of a frame next to each other changes VideoMAE's output, whatever the frame shows** |

Accuracy with the same content (500 videos):

| k different frames | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| as a short clip of k frames (%) | 42.4 | 56.2 | 63.2 | 66.4 | 68.4 | 73.6 | 74.8 | 76.0 |
| repeated to 16 frames (%) | 40.4 | 43.0 | 53.4 | 45.6 | 67.2 | 66.8 | 74.4 | 76.0 |
| pairs with two identical frames (repeated clip) | 8 of 8 | 8 of 8 | 6 of 8 | 8 of 8 | 4 of 8 | 4 of 8 | 2 of 8 | 0 of 8 |

The drop follows the number of identical pairs, not k. For example, k = 8 (every pair identical) is worse than k = 6 (6 of 8 identical). For comparison, V-JEPA2 also reads frames in pairs, but loses only 2.2 points at k = 8.

## 3. Reallocation: the reference clip isn't normal for the model

**Design.** We pick 8 frames (every other one of 16) and show each twice, in neighbouring slots (m = 2). Copies of the target take slots from other frames. Removed frames are filled in (freeze). This is the design we use for R3D.

**Check.** We built the clips for 50 videos, keeping the ones the model gets wrong too (43 kept; `videomae_diag/`). We compared the reference clip with the original 16 different frames.

| measure | VideoMAE | R3D-50 | TRN |
|---|---|---|---|
| same prediction, original vs reference | **37.2%** | 94.1% | 87.8% |
| change in the probability of the original prediction | **−0.36** | −0.001 | −0.047 |
| accuracy, original → reference | 79% → 37% | – | – |

In this reference, every pair consists of two identical frames. So the reference changes VideoMAE's prediction in most videos. It isn't a fair stand-in for how the model behaves on real clips.

## 4. Replacement: the model's importance of a duplicated frame turns negative

We designed a third option for VideoMAE (`design="replace"`), which avoids an all-identical reference:
- **Reference:** the original 16 different frames (m = 1).
- **Target clip:** the target fills a block of m neighbouring slots, aligned with the frame pairs. It replaces the m − 1 frames that were there.
- **Control clip:** the same slots are filled with the least important frame (the *recipient*). The target keeps one slot.
- **Removing frames:** deletion.

**Check.** We built the clips for 50 videos (33 kept, 57 targets; `videomae_replace_diag/`).

**The prediction stays the same in most clips:**

| m | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| target clip: same prediction (%) | 96.4 | 96.2 | 85.7 | 86.7 |
| control clip: same prediction (%) | 100 | 92.5 | 89.8 | 84.4 |
| target–control pairs that pass the gate | 53 of 55 | 49 of 53 | 40 of 49 | 36 of 45 |

**But the model's importance of a duplicated frame becomes negative.** A frame's importance I is the drop in the predicted class's probability when every copy of it is deleted. A negative value means deleting it *raises* the probability. Medians, exact copies:

| m | I of the target, reference | I of the target, target clip | I of the recipient, reference | I of the recipient, control clip | targets with negative I in the target clip |
|---|---|---|---|---|---|
| 2 | +0.052 | −0.011 | +0.001 | −0.020 | 60% |
| 4 | +0.063 | −0.028 | +0.001 | −0.033 | 60% |
| 6 | +0.065 | −0.058 | +0.001 | −0.049 | 69% |
| 8 | +0.066 | −0.062 | +0.001 | −0.083 | 78% |

What this shows:
- **Deleting the neighbouring copies of any frame raises the predicted class's probability**, whether it's the important target or the unimportant recipient. The effect grows with m.
- **Once the target is duplicated, the model treats it like the recipient.** In the reference, the target is 50–65 times more important than the recipient.
- **This isn't caused by deletion itself.** When a frame is deleted, the frames after it are paired differently. But in the reference, a single frame's importance doesn't depend on its position in the clip (correlation with position: −0.025). So the re-pairing doesn't distort the measure.

## 5. Conclusion

In VideoMAE, neighbouring copies of a frame form pairs of identical frames, and the model reacts to those pairs rather than to what the frame shows. So duplicating a frame changes how much the model relies on it:
- **with reallocation,** the reference already changes the prediction in 63% of videos;
- **with replacement,** the model's importance of any duplicated frame turns negative.

If a method gave duplicated frames less credit on VideoMAE, that could be correct: the model really does rely on them less. So E1 can't tell method errors from correct answers on this model. **We report VideoMAE as a model where E1's starting assumption doesn't hold.**

One option we didn't pursue: duplicate whole frame pairs instead of single frames, so every pair keeps two different neighbouring frames. That avoids identical pairs, but it changes what is being duplicated, from frames to pairs.

## 6. Files and how to reproduce

Files in `results/e1/`:

| file or folder | what it contains |
|---|---|
| `videomae_diag/` | the reallocation check: clips (reference only, `--ms 2 --include-wrong`) and gate |
| `videomae_replace_diag/` | the replacement check: clips for m = 2, 4, 6, 8 and gate (run `e1_metrics.py gate` to rebuild the gate file) |
| `videomae_replace_diag.log` | build log |
| `../frame_count_videomae_*.csv` | accuracy against clip length, 2–16 frames |
| `../length_fixed_content_videomae_*.csv` | accuracy for k frames as a short clip vs repeated to 16 |

Commands:
```
python eval_frame_count.py --models videomae --frames 2 4 6 8 10 12 14 16 --limit 500 --datasets videomae=ucf101_test --out results/frame_count_videomae
python eval_length_fixed_content.py --models videomae --datasets videomae=ucf101_test --ks 2 4 6 8 10 12 14 16 --limit 500 --out results/length_fixed_content_videomae

# reallocation check (needs design="realloc", removal="late" for videomae in MODEL_SPECS)
python motivation/e1_build_conditions.py --model videomae --limit 50 --ms 2 --dup-types exact --include-wrong --out results/e1/videomae_diag
python motivation/e1_metrics.py gate --model videomae --out results/e1/videomae_diag

# replacement check (current MODEL_SPECS: design="replace", removal="drop")
python motivation/e1_build_conditions.py --model videomae --limit 50 --fp16 --out results/e1/videomae_replace_diag
python motivation/e1_metrics.py gate --model videomae --out results/e1/videomae_replace_diag
```
