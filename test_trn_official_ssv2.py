"""
test_trn_official_ssv2.py -- accuracy of the ORIGINAL authors' pretrained TRN
(TRNmultiscale, BN-Inception, trained on Something-Something v1) on your local SSv2
test set. Fallback for test_trn_ssv2.py now that play-fair's own checkpoint mirror is
dead -- see models/trn_official.py and download_trn_official.py for why.

Setup (once):
    python play-fair/checkpoints/download_trn_official.py

Run:
    python test_trn_official_ssv2.py
"""
import torch

from dataloaders import ssv2
from models.trn_official import TRNOfficial


def test_s2s():
    model = TRNOfficial()
    class_names = list(model.label2id.keys())

    d_names, paths = ssv2.get_ssv2_paths()
    n_files = len(paths)

    missing = [c for c in class_names if c not in d_names]
    if missing:
        print(f"[warn] {len(missing)}/{len(class_names)} model class names have no matching "
              f"folder under CONST.SSV2_PATH -- label mapping may be off. First few: {missing[:5]}")

    n_correct = 0
    n_samples = 0
    for idx, p in enumerate(paths):
        if idx > 0:
            print(f'{idx / n_files * 100:.2f} % is done. Running acc: {n_correct / n_samples * 100:.2f} %',
                  end='\r')
        if d_names[idx] not in model.label2id:
            continue  # class name mismatch -- see the warning above
        gt_idx = model.label2id[d_names[idx]]
        with torch.no_grad():
            pred_cls = model.predict_from_path(p)
        if pred_cls == gt_idx:
            n_correct += 1
        n_samples += 1

    print(f'\nAccuracy = {n_correct / n_samples * 100:.2f} %  ({n_correct}/{n_samples})')
    return n_correct / n_samples


if __name__ == '__main__':
    test_s2s()
