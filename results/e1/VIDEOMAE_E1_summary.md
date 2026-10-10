# E1 summary: VideoMAE (ViT-B, UCF101) — excluded from the E1 test

**Status:** VideoMAE is **not used** for the E1 test. Under every duplication design that its input format allows, duplicating a frame changes the model's dependence on that content. E1 requires that dependence to stay approximately unchanged, so that a change in a duplicated frame's attribution can be interpreted as an error of the attribution method. This document records the evidence.

**Model:** `nateraw/videomae-base-finetuned-ucf101`, 16 frames at 224 px, tubelet = 2 (each pair of consecutive frames is embedded jointly), fixed sinusoidal position table of 1,568 tokens. The checkpoint's q/v attention biases are restored by `models/videomae_ucf101.py`, because `transformers` 5.8.0.dev0 doesn't load them. **Data:** UCF101 split-1 test set (`ucf101_test`).

## 1. Input format

| property | measurement | consequence |
|---|---|---|
| inputs longer than 16 frames | `RuntimeError` at 18 and 24 frames (more tokens than position-table rows) | the insertion design is impossible |
| inputs shorter than 16 frames | run with the table sliced to the first T/2 time positions; accuracy rises smoothly from 42.4% (2 frames) to 76.0% (16 frames), with no step change (500 videos) | deletion is possible |
| input length at fixed content | the k-frame input is at least as accurate as the same k frames repeated to 16 frames, for every k (500 videos) | no length penalty of R3D's kind |
| frame pairs of two identical frames | accuracy of the repeated input falls in proportion to the number of such pairs: −13.2 pp (k = 4) and −20.8 pp (k = 8), where every pair consists of identical frames; −0.4 to −6.8 pp where half or fewer do | **duplicating frames into consecutive slots changes VideoMAE's output, independently of their content** |

Accuracy at fixed content (500 videos):

| k distinct frames | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| k-frame input (%) | 42.4 | 56.2 | 63.2 | 66.4 | 68.4 | 73.6 | 74.8 | 76.0 |
| repeated to 16 (%) | 40.4 | 43.0 | 53.4 | 45.6 | 67.2 | 66.8 | 74.4 | 76.0 |
| frame pairs of identical frames (repeated input) | 8 / 8 | 8 / 8 | 6 / 8 | 8 / 8 | 4 / 8 | 4 / 8 | 2 / 8 | 0 / 8 |

For comparison, V-JEPA2, which also embeds frame pairs, loses only 2.2 pp at k = 8.

## 2. Reallocation design: the reference is not representative

**Design:** K = 8 contents (every other sampled frame), each in 2 consecutive slots (m_ref = 2); copies of the target take slots from other contents; frames removed by freeze.

**Check:** a condition build on 50 videos with `--include-wrong` (43 kept; `videomae_diag/`) compared the reference with the original 16 distinct frames.

| measure | VideoMAE | R3D | TRN |
|---|---|---|---|
| same predicted class, original vs reference | **37.2%** | 94.1% | 87.8% |
| mean change in P(original prediction) | **−0.36** | −0.001 | −0.047 |
| accuracy, original → reference | 79% → 37% | – | – |

Every frame pair in this reference consists of two identical frames. The reference therefore changes VideoMAE's prediction in most videos and doesn't represent its behaviour on real clips.

## 3. Replacement design: the model's dependence on duplicated content changes sign

**Design** (`design="replace"`, implemented for VideoMAE):
- **reference:** the original 16 distinct frames (m_ref = 1);
- **target condition:** the target t fills a pair-aligned block of m consecutive slots, replacing m − 1 neighbouring frames;
- **control:** the same slots are filled with the least important content (the recipient), and t keeps one slot;
- **removal:** deletion.

**Check:** a condition build on 50 videos (33 kept, 57 targets; `videomae_replace_diag/`).

**The prediction is preserved:**

| m | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| target agreement (%) | 96.4 | 96.2 | 85.7 | 86.7 |
| control agreement (%) | 100 | 92.5 | 89.8 | 84.4 |
| target–control pairs passing the gate | 53 / 55 | 49 / 53 | 40 / 49 | 36 / 45 |

**But the model-based importance of duplicated content becomes negative.** I(c) is the decrease in P(predicted class) when every copy of c is deleted (medians, exact copies):

| m | I(t), reference | I(t), target condition | I(recipient), reference | I(recipient), control condition | targets with I(t) < 0 in the target condition |
|---|---|---|---|---|---|
| 2 | +0.052 | −0.011 | +0.001 | −0.020 | 60% |
| 4 | +0.063 | −0.028 | +0.001 | −0.033 | 60% |
| 6 | +0.065 | −0.058 | +0.001 | −0.049 | 69% |
| 8 | +0.066 | −0.062 | +0.001 | −0.083 | 78% |

- **Deleting the consecutive copies of any content raises the predicted-class probability,** whether that content is the important target or the unimportant recipient. The effect grows with m.
- **When t is duplicated, its importance is no longer distinguishable from the recipient's,** although in the reference it is about 50–65 times larger.
- **This isn't an artefact of deletion.** Single-frame importances in the reference don't depend on slot position (correlation with slot index −0.025), so re-pairing the frames after a deleted frame doesn't distort the measurement.

## Conclusion

In VideoMAE, consecutive copies of a frame form pairs of identical frames, and the model's output responds to those pairs rather than to the duplicated content. Its dependence on a duplicated content therefore isn't preserved:
- **in the reallocation design,** the reference already changes the prediction in 63% of videos;
- **in the replacement design,** the model-based importance of duplicated content becomes negative.

A reduced attribution for duplicated frames would be consistent with the model's behaviour, so E1 can't distinguish attribution errors from correct attributions on this model. **VideoMAE is reported as a model for which the E1 premise doesn't hold.**

Duplicating whole frame pairs, so that every pair keeps two different, consecutive frames, would avoid identical-frame pairs. It changes the unit of duplication from frames to pairs and wasn't pursued.

## Files (`results/e1/`)

| file / folder | contents |
|---|---|
| `videomae_diag/` | reallocation-design check: conditions (reference only, `--ms 2 --include-wrong`) and gate |
| `videomae_replace_diag/` | replacement-design check: conditions for m = 2, 4, 6, 8 and gate (run `e1_metrics.py gate` to regenerate the gate CSV) |
| `videomae_replace_diag.log` | build log |
| `../frame_count_videomae_*.csv` | accuracy against input length, 2–16 frames |
| `../length_fixed_content_videomae_*.csv` | accuracy at fixed content |

## Reproduce

```
python eval_frame_count.py --models videomae --frames 2 4 6 8 10 12 14 16 --limit 500 --datasets videomae=ucf101_test --out results/frame_count_videomae
python eval_length_fixed_content.py --models videomae --datasets videomae=ucf101_test --ks 2 4 6 8 10 12 14 16 --limit 500 --out results/length_fixed_content_videomae

# reallocation check (requires design="realloc", removal="late" for videomae in MODEL_SPECS)
python motivation/e1_build_conditions.py --model videomae --limit 50 --ms 2 --dup-types exact --include-wrong --out results/e1/videomae_diag
python motivation/e1_metrics.py gate --model videomae --out results/e1/videomae_diag

# replacement check (current MODEL_SPECS: design="replace", removal="drop")
python motivation/e1_build_conditions.py --model videomae --limit 50 --fp16 --out results/e1/videomae_replace_diag
python motivation/e1_metrics.py gate --model videomae --out results/e1/videomae_replace_diag
```
