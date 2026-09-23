"""
test_trn_ssv2.py -- accuracy of the pretrained Multiscale TRN (BN-Inception, SSv2) on your
local SSv2 test set (CONST.SSV2_PATH), mirroring test_ssv2.py's test_s2s() for VJEPA2 so the
numbers are directly comparable.

Setup (once):
    python play-fair/checkpoints/download_trn.py

Run:
    python test_trn_ssv2.py
"""
import torch

from dataloaders import ssv2
from models.trn import TRN


def test_s2s():
    model = TRN()
    class_names = list(model.label2id.keys())

    d_names, paths = ssv2.get_ssv2_paths()
    n_files = len(paths)

    # make sure all the class names are present in the list of dirs
    for c in class_names:
        assert c in d_names, f'{c} is not in the list of dirs'

    n_correct = 0
    n_samples = 0
    for idx, p in enumerate(paths):
        if idx > 0:
            print(f'{idx / n_files * 100:.2f} % is done. Running acc: {n_correct / n_samples * 100:.2f} %',
                  end='\r')
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
