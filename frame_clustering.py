"""
frame_clustering.py -- group temporally adjacent, visually similar frames within a clip.

Embeds each sampled frame with models/dinov2.py's DINOv2 encoder, then runs agglomerative
clustering constrained to only ever merge index-adjacent clusters (a band connectivity
matrix), so every cut of the resulting tree is a valid contiguous segmentation of the clip --
from n_frames singleton frames (finest) up to one cluster covering the whole clip (coarsest).

    Z, embeddings = frame_hierarchy("some_video.mp4")
    from scipy.cluster.hierarchy import dendrogram, cut_tree
    dendrogram(Z)                     # visualize the merge tree
    cut_tree(Z, n_clusters=[3, 5])    # frame->cluster labels at chosen granularities
"""
import numpy as np
from scipy.cluster.hierarchy import cut_tree
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering

from models.dinov2 import DINOv2
from models.video_utils import sample_frames

_dino = None


def _get_dino() -> DINOv2:
    global _dino
    if _dino is None:
        _dino = DINOv2()
    return _dino


def _adjacent_connectivity(n: int) -> csr_matrix:
    """(n,n) sparse matrix connecting each frame index i only to i-1 and i+1."""
    rows = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def frame_hierarchy(video_path, n_frames: int = 16, linkage: str = "ward"):
    """video_path -> (Z, embeddings).

    Z is a (n_frames-1, 4) scipy linkage matrix [merge_a, merge_b, distance, count]: the full
    merge tree from connectivity-constrained agglomerative clustering (adjacent frames only),
    ordered finest -> coarsest. Feed it to scipy.cluster.hierarchy for cutting/plotting.
    embeddings is the (n_frames, D) per-frame DINOv2 embedding tensor (raw, un-normalized).

    Clustering runs on L2-normalized embeddings with euclidean distance rather than cosine
    directly, since for unit-norm vectors euclidean distance is a monotonic function of cosine
    similarity (||a-b||^2 = 2 - 2*cos_sim) -- this makes the default "ward" linkage usable
    (sklearn restricts it to euclidean) while still ranking frame pairs by cosine similarity.
    """
    frames = sample_frames(video_path, n_frames)
    embeddings = _get_dino().embed_frames(frames)
    n = len(embeddings)

    normalized = embeddings.numpy()
    normalized = normalized / np.linalg.norm(normalized, axis=1, keepdims=True)

    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0,  # forces the full tree, no early stop
        connectivity=_adjacent_connectivity(n),
        linkage=linkage, metric="euclidean", compute_full_tree=True,
    ).fit(normalized)

    # scipy's linkage format needs a running leaf-count per merge, which sklearn doesn't
    # expose directly -- accumulate it from children_ (standard sklearn->scipy conversion).
    counts = np.zeros(clustering.children_.shape[0])
    for i, (a, b) in enumerate(clustering.children_):
        counts[i] = sum(1 if child < n else counts[child - n] for child in (a, b))
    Z = np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)
    return Z, embeddings


def hierarchy_groups(Z, n_frames: int) -> dict:
    """Z -> {k: [[frame indices], ...]} for every level from k=n_frames (finest, all
    singletons) down to k=1 (coarsest, one group spanning the whole clip). Each level's groups
    are ordered left-to-right and always contiguous (e.g. [0, 1, 2], never [0, 2, 5]), since
    clustering only ever merges index-adjacent frames."""
    all_k = list(range(n_frames, 0, -1))
    labels = cut_tree(Z, n_clusters=all_k)  # (n_frames, len(all_k))

    groups_by_k = {}
    for col, k in enumerate(all_k):
        groups = {}
        for frame_idx, lbl in enumerate(labels[:, col]):
            groups.setdefault(int(lbl), []).append(frame_idx)
        groups_by_k[k] = sorted(groups.values(), key=lambda g: g[0])
    return groups_by_k


def print_hierarchy(Z, n_frames: int) -> None:
    """Pretty-print hierarchy_groups(), one line per level, finest to coarsest."""
    for k, groups in hierarchy_groups(Z, n_frames).items():
        print(f"k={k:2d}: " + " | ".join(str(g) for g in groups))


if __name__ == "__main__":
    Z, embeddings = frame_hierarchy(
        r"D:\datasets\SSV2\s2s_test\Attaching something to something\5289.webm")
    print_hierarchy(Z, len(embeddings))


