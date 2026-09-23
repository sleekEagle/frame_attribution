"""
eval_accuracy.py -- generic top-1 accuracy eval for any (model, dataset) pair, replacing the
old per-combination scripts (test_ssv2.py, test_trn_ssv2.py, test_trn_official_ssv2.py,
test_ucf101.py). Model and dataset loading goes through models/registry.py and
dataloaders/registry.py, which is what keeps this script model/dataset-agnostic.

    python eval_accuracy.py --model vjepa2                  # ssv2, its default dataset
    python eval_accuracy.py --model trn
    python eval_accuracy.py --model trn_official
    python eval_accuracy.py --model r3d                      # ucf101, its default dataset
    python eval_accuracy.py --model videomae                 # ucf101, different architecture
    python eval_accuracy.py --model trn_official --dataset ssv2_sampled

--dataset defaults to models.registry.MODEL_DATASET[--model] and only needs overriding for an
alternative loader of the same dataset (e.g. ssv2_sampled).
"""
import argparse

import torch

from dataloaders.registry import get_dataloader
from models.registry import MODEL_DATASET, get_model


def evaluate(model_name: str, dataset_name: str = None):
    dataset_name = dataset_name or MODEL_DATASET[model_name]
    model = get_model(model_name)
    d_names, paths = get_dataloader(dataset_name)
    n_files = len(paths)

    missing = sorted({n for n in d_names if n not in model.label2id})


    if missing:
        print(f"[warn] {len(missing)} dataset class name(s) have no matching entry in "
              f"{model_name}.label2id -- label mapping may be off. First few: {missing[:5]}")

    n_correct = 0
    n_samples = 0
    for idx, p in enumerate(paths):
        if idx > 0:
            print(f'{idx / n_files * 100:.2f} % is done. Running acc: '
                  f'{n_correct / max(n_samples, 1) * 100:.2f} %', end='\r')
        if d_names[idx] not in model.label2id:
            continue
        gt_idx = model.label2id[d_names[idx]]
        with torch.no_grad():
            pred_cls = model.predict_from_path(p)
        if pred_cls is None:  # model couldn't build a clip from this path (e.g. too few frames)
            continue
        if pred_cls == gt_idx:
            n_correct += 1
        n_samples += 1

    acc = n_correct / n_samples * 100
    print(f'\nAccuracy = {acc:.2f} %  ({n_correct}/{n_samples})')
    return acc
'''
UCF101, R3D: Accuracy = 90.54 %  (12060/13320)
        videoMAE: Accuracy = 87.69 %  (11680/13320)

trn_official, SSV2: Accuracy = 33.58 %  (9118/27157)

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = 'trn_official', choices=sorted(MODEL_DATASET))
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()
    evaluate(args.model, args.dataset)
