"""
dataloaders/registry.py -- single entry point for loading path lists for any dataset used in
this repo, so scripts don't need per-dataset imports:

    d_names, paths = get_dataloader("ssv2")     # or "ucf101"

Return shape is uniform: d_names[i] is the ground-truth class name (str) for paths[i]. Look up
its index with model.label2id[d_names[i]] (see models/registry.py's get_model()), then feed
paths[i] to model.predict_from_path(paths[i]).
"""
from dataloaders import ssv2
from dataloaders import ucf101_loader

_REGISTRY = {
    "ssv2": ssv2.get_ssv2_paths,
    "ssv2_sampled": ssv2.get_sampled_paths,  # returns (cls_list, path_list) -- same shape
    "ucf101": ucf101_loader.get_ucf101_paths,  # full dataset, train+test combined
    "ucf101_test": ucf101_loader.get_ucf101_test_paths,  # split 1 test set only (eval)
}


def get_dataloader(name: str, **kwargs):
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
