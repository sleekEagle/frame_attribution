"""
models/registry.py -- single entry point for loading any of this repo's video-classification
models with a uniform interface, so scripts don't need per-model imports:

    model = get_model("vjepa2")   # or "trn", "trn_official", "r3d"
    model.label2id                        class name (str) -> class index (int)
    model.predict_from_path(path)         -> predicted class index (int)
    model.predict_from_batch_path(paths)  -> (N, num_classes) logits tensor

"path" means a video file for vjepa2/trn/trn_official, and a directory of extracted jpg
frames for r3d (see models/r3d_ucf101.py) -- matches how each model's data was prepared.

Construction is lazy and expensive (weights are loaded from disk/HF), so instances are cached
by name; pass cache=False or any extra kwarg to force a fresh instance.
"""
from models.ssv2 import VJEPA2
from models.trn import TRN
from models.trn_official import TRNOfficial
from models.r3d_ucf101 import R3DUCF101
from models.videomae_ucf101 import VideoMAEUCF101

_REGISTRY = {
    "vjepa2": VJEPA2,
    "trn": TRN,
    "trn_official": TRNOfficial,
    "r3d": R3DUCF101,
    "videomae": VideoMAEUCF101,
}

# the dataset each model is evaluated on -- lets get_dataloader() be inferred from the model
# name alone (see eval_accuracy.py).
MODEL_DATASET = {
    "vjepa2": "ssv2",
    "trn": "ssv2",
    "trn_official": "ssv2",
    "r3d": "ucf101",
    "videomae": "ucf101",
}

_cache = {}


def get_model(name: str, cache: bool = True, **kwargs):
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_REGISTRY)}")
    if cache and not kwargs and key in _cache:
        return _cache[key]
    model = _REGISTRY[key](**kwargs)
    if hasattr(model, "eval"):
        model.eval()
    if cache and not kwargs:
        _cache[key] = model
    return model
