"""
def_calibration/defocus_calib.py -- standalone edge-spread-function extraction for ChArUco
calibration images.

extract_esf_samples() measures the edge-spread function (ESF) directly in image pixels along
the boundary between adjacent ChArUco squares, together with the exact geometric depth (from
the board's PnP pose) of the point where each scan line crosses that boundary. Unlike Laplacian
variance, the ESF's width is provably linear in the circle-of-confusion diameter (see the
defocus-blur theory in focaldist_estimation.py's module docstring), so it is a lower-noise blur
proxy for D_focus estimation.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import cv2.aruco as aruco
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _bilinear_sample(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate grayscale intensity at fractional pixel coords (xs, ys). Points
    that fall outside the image come back as NaN so the caller can drop that whole line."""
    h, w = gray.shape
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    out = np.full(xs.shape, np.nan)
    x0 = np.floor(xs[valid]).astype(int)
    y0 = np.floor(ys[valid]).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = xs[valid] - x0
    fy = ys[valid] - y0
    out[valid] = (gray[y0, x0] * (1 - fx) * (1 - fy) + gray[y0, x1] * fx * (1 - fy) +
                  gray[y1, x0] * (1 - fx) * fy + gray[y1, x1] * fx * fy)
    return out


def extract_esf_samples(image_path: Path, detector, obj_points_all, K, D, info: dict,
                         lines_per_square: int = 3, margin_frac: float = 0.1,
                         n_obj_samples: int = 300, sample_spacing_px: float = 0.5,
                         min_contrast: float = 15.0):
    """One calibration image -> list of edge-spread-function samples, one per (square, edge
    side, scan-line height). For each square boundary, a horizontal object-space segment
    straddling it is projected into the image through the board's PnP pose (K/D are applied via
    cv2.projectPoints, so the raw distorted image itself is never resampled/undistorted -- that
    would blur it and corrupt the very thing we're trying to measure). The projected polyline is
    then resampled onto a uniform *pixel*-arc-length grid (not a uniform object-space grid,
    since perspective makes the two spacings differ) and bilinearly sampled for intensity, which
    gives the edge-spread function directly in image pixels, centered on the edge crossing.
    Depth is the Z-coordinate (camera coordinates) of the object-space point where the scan
    line crosses the edge.

    detector: cv2.aruco.CharucoDetector for the board.
    obj_points_all: board.getChessboardCorners(), indexed by ChArUco corner id.
    K, D: that focal length's calibrated camera matrix and distortion coefficients.
    info: pattern_info_charuco.json's "charuco" section (squares_x, squares_y,
        square_length_m, ...).

    Each returned dict also carries "img_xy" (Nx2), the actual image pixel coordinates the
    ESF was sampled at, for overlaying on the source image as a sanity check. "scan" is
    "horizontal" for a horizontal scan line crossing one of the square's left/right (vertical)
    edges, or "vertical" for a vertical scan line crossing one of its top/bottom (horizontal)
    edges -- sampling both doubles coverage and lets horizontal- vs vertical-edge blur be
    compared separately.
    """
    gray_u8 = np.asarray(Image.open(image_path).convert("L"))

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray_u8)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 6:
        return []
    obj_points = obj_points_all[charuco_ids.flatten()]
    img_points = charuco_corners.reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
    if not ok:
        return []
    R, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape(3, 1)

    gray = gray_u8.astype(np.float64)
    sq = info["square_length_m"]
    squares_x, squares_y = info["squares_x"], info["squares_y"]
    margin = margin_frac * sq
    t = np.linspace(-margin, margin, n_obj_samples)
    fracs = (np.arange(lines_per_square) + 1) / (lines_per_square + 1)  # e.g. .25, .5, .75

    def sample_scan_line(obj_pts):
        """obj_pts: Nx3 object-space line straddling an edge at t=0 (the middle sample) ->
        (s, esf, px, py, contrast), or None if it fails the coverage/contrast checks."""
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        img_pts = img_pts.reshape(-1, 2)

        seg_len = np.linalg.norm(np.diff(img_pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_len)])
        if arc[-1] < 2 * sample_spacing_px:
            return None
        s_edge = np.interp(0.0, t, arc)

        s_uniform = np.arange(0.0, arc[-1], sample_spacing_px)
        px = np.interp(s_uniform, arc, img_pts[:, 0])
        py = np.interp(s_uniform, arc, img_pts[:, 1])

        esf = _bilinear_sample(gray, px, py)
        if np.isnan(esf).any():
            return None

        k = max(3, len(esf) // 10)
        contrast = abs(float(esf[:k].mean()) - float(esf[-k:].mean()))
        if contrast < min_contrast:
            return None
        return s_uniform - s_edge, esf, px, py, contrast

    results = []
    for row in range(squares_y):
        for col in range(squares_x):
            # horizontal scan lines: cross the square's left/right (vertical) edges
            for edge, x_edge in (("left", col * sq), ("right", (col + 1) * sq)):
                if (edge == "left" and col == 0) or (edge == "right" and col == squares_x - 1):
                    continue
                for frac in fracs:
                    y_line = (row + frac) * sq
                    obj_pts = np.stack(
                        [x_edge + t, np.full_like(t, y_line), np.zeros_like(t)], axis=1)
                    sampled = sample_scan_line(obj_pts)
                    if sampled is None:
                        continue
                    s, esf, px, py, contrast = sampled

                    obj_edge = np.array([[x_edge, y_line, 0.0]])
                    depth = float((R @ obj_edge.T + tvec)[2, 0])
                    if depth <= 0:
                        continue

                    results.append({
                        "row": row, "col": col, "edge": edge, "scan": "horizontal",
                        "frac": float(frac), "depth": depth, "s": s, "esf": esf,
                        "contrast": contrast, "img_xy": np.stack([px, py], axis=1),
                    })

            # vertical scan lines: cross the square's top/bottom (horizontal) edges
            for edge, y_edge in (("top", row * sq), ("bottom", (row + 1) * sq)):
                if (edge == "top" and row == 0) or (edge == "bottom" and row == squares_y - 1):
                    continue
                for frac in fracs:
                    x_line = (col + frac) * sq
                    obj_pts = np.stack(
                        [np.full_like(t, x_line), y_edge + t, np.zeros_like(t)], axis=1)
                    sampled = sample_scan_line(obj_pts)
                    if sampled is None:
                        continue
                    s, esf, px, py, contrast = sampled

                    obj_edge = np.array([[x_line, y_edge, 0.0]])
                    depth = float((R @ obj_edge.T + tvec)[2, 0])
                    if depth <= 0:
                        continue

                    results.append({
                        "row": row, "col": col, "edge": edge, "scan": "vertical",
                        "frac": float(frac), "depth": depth, "s": s, "esf": esf,
                        "contrast": contrast, "img_xy": np.stack([px, py], axis=1),
                    })
    return results


def esf_width_10_90(s: np.ndarray, esf: np.ndarray, plateau_frac: float = 0.1):
    """10%-90% rise/fall width of one ESF curve (same units as `s` -- pixels, here). For a
    Gaussian PSF this equals exactly 2*Phi^-1(0.9)*sigma ~= 2.563*sigma (see this module's
    docstring), so it's a low-noise, linear proxy for blur scale. Returns None if the two
    plateaus can't be told apart (e.g. a flat/degenerate curve).

    Thresholds are set relative to the curve's own two plateau levels (averaged over
    `plateau_frac` of each end, same convention as the contrast check in
    extract_esf_samples()), not its raw min/max, so a single noisy sample can't skew them. The
    affine normalization below maps the left end to 0 and the right end to 1 regardless of
    whether the edge is rising or falling, so no direction handling is needed.
    """
    n = len(esf)
    k = max(3, int(n * plateau_frac))
    lo, hi = float(esf[:k].mean()), float(esf[-k:].mean())

    if hi == lo:
        return None

    norm = (esf - lo) / (hi - lo)
    s10 = float(np.interp(0.1, norm, s))
    s90 = float(np.interp(0.9, norm, s))
    return abs(s90 - s10)


def process_directory(image_dir: Path, detector, obj_points_all, K, D, info: dict, **kwargs):
    """Run extract_esf_samples() over every *.JPG in image_dir and reduce each sample to its
    10-90 width, pooling results across the whole directory (kwargs are forwarded to
    extract_esf_samples(), e.g. margin_frac). Returns a list of lightweight dicts (no s/esf/
    img_xy arrays, just the numbers needed for a D_focus fit): {"image", "depth", "width",
    "row", "col", "edge", "scan"}."""
    results = []
    for image_path in sorted(image_dir.glob("*.JPG")):
        for s in extract_esf_samples(image_path, detector, obj_points_all, K, D, info, **kwargs):
            width = esf_width_10_90(s["s"], s["esf"])
            if width is None:
                continue
            results.append({
                "image": image_path.name, "depth": s["depth"], "width": width,
                "row": s["row"], "col": s["col"], "edge": s["edge"], "scan": s["scan"],
            })
    return results


def _load_setup(focal_name: str, camera_dir: Path, calib_root: Path):
    """Load the board geometry/detector and that focal length's intrinsics."""
    with open(calib_root / "pattern_info_charuco.json") as f:
        info = json.load(f)["charuco"]
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, info["dictionary"]))
    board = aruco.CharucoBoard((info["squares_x"], info["squares_y"]),
                                info["square_length_m"], info["marker_length_m"], dictionary)
    obj_points_all = board.getChessboardCorners()
    detector = aruco.CharucoDetector(board)
    npz = np.load(camera_dir / "calibration" / f"{focal_name}.npz")
    return detector, obj_points_all, npz["K"], npz["D"], info


def main():
    """Batch-process every image in a directory (e.g. an fl_XXmm folder): extract ESF samples,
    reduce each to a 10-90 width, and write out a pooled (depth, width) CSV plus a width-vs-depth
    scatter plot. Also saves points-on-image and ESF-curve diagnostic plots for one
    representative image, as a sanity check. image_dir is expected under the usual
    <calib_root>/<camera_dir>/fl_<X>mm/*.JPG layout (see focaldist_estimation_calibimgs.py's
    module docstring), so calib_root and camera_dir are inferred from image_dir's own parent
    directories."""
    default_dir = (r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco"
                    r"\Global_calibration_set\ChArUco_pattern\EOS_6D_A\fl_32mm")
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("image_dir", type=Path, nargs="?", default=Path(default_dir),
                         help="directory of calibration images to process (e.g. an fl_XXmm folder)")
    parser.add_argument("--out_plot", type=Path, default=r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco\Global_calibration_set\ChArUco_pattern\EOS_6D_A\debug",
                         help="where to save plots / the widths CSV")
    parser.add_argument("--n_plot", type=int, default=6, help="how many samples to plot")
    args = parser.parse_args()

    image_dir = args.image_dir
    focal_name = image_dir.name  # e.g. "fl_28mm"
    camera_dir = image_dir.parent  # e.g. .../EOS_6D_A
    calib_root = camera_dir.parent  # .../ChArUco_pattern
    detector, obj_points_all, K, D, info = _load_setup(focal_name, camera_dir, calib_root)

    out_dir = args.out_plot or image_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = process_directory(image_dir, detector, obj_points_all, K, D, info)
    print(f"{image_dir}: {len(results)} (depth, width) samples")
    if not results:
        return
    for r in results:
        print(f"{r['depth']:.4f}\t{r['width']:.3f}\t{r['image']}")

    csv_path = out_dir / f"{focal_name}_widths.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "row", "col", "edge", "scan", "depth", "width"])
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([r["depth"] for r in results], [r["width"] for r in results], s=4, alpha=0.3)
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("10-90 width (px)")
    ax.set_title(f"width vs depth: {focal_name}")
    fig.tight_layout()
    width_plot_path = out_dir / f"{focal_name}_width_vs_depth.png"
    fig.savefig(width_plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote {width_plot_path}")

    # horizontal- vs vertical-scan bias check: depth and width distributions side by side
    horiz = [r for r in results if r["scan"] == "horizontal"]
    vert = [r for r in results if r["scan"] == "vertical"]
    fig, (ax_d, ax_w) = plt.subplots(1, 2, figsize=(10, 5))
    ax_d.violinplot([[r["depth"] for r in horiz], [r["depth"] for r in vert]], showmeans=True)
    ax_d.set_xticks([1, 2], labels=["horizontal", "vertical"])
    ax_d.set_ylabel("depth (m)")
    ax_d.set_title("depth by scan direction")
    ax_w.violinplot([[r["width"] for r in horiz], [r["width"] for r in vert]], showmeans=True)
    ax_w.set_xticks([1, 2], labels=["horizontal", "vertical"])
    ax_w.set_ylabel("10-90 width (px)")
    ax_w.set_title("width by scan direction")
    fig.suptitle(f"horizontal vs vertical scan bias: {focal_name}")
    fig.tight_layout()
    bias_plot_path = out_dir / f"{focal_name}_scan_bias.png"
    fig.savefig(bias_plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote {bias_plot_path}")

    # points-on-image + ESF-curve diagnostic plots, for one representative image
    image_path = sorted(image_dir.glob("*.JPG"))[0]
    samples = extract_esf_samples(image_path, detector, obj_points_all, K, D, info)
    if not samples:
        return

    gray_u8 = np.asarray(Image.open(image_path).convert("L"))
    points_path = out_dir / f"{image_path.stem}_points.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(gray_u8, cmap="gray", vmin=0, vmax=255)
    scan_colors = {"horizontal": "tab:orange", "vertical": "tab:cyan"}
    edge_xy = np.empty((len(samples), 2))
    labeled_scans = set()
    for i, s in enumerate(samples):
        xy = s["img_xy"]
        label = f"{s['scan']} scan" if s["scan"] not in labeled_scans else None
        labeled_scans.add(s["scan"])
        ax.plot(xy[:, 0], xy[:, 1], linewidth=0.6, alpha=0.6, color=scan_colors[s["scan"]],
                label=label)
        edge_xy[i] = np.interp(0.0, s["s"], xy[:, 0]), np.interp(0.0, s["s"], xy[:, 1])
    ax.scatter(edge_xy[:, 0], edge_xy[:, 1], s=2, color="red", zorder=3,
               label="edge crossing (depth sample)")
    ax.set_title(f"scan lines considered: {image_path.name} ({len(samples)} samples)")
    ax.legend(fontsize=8, loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(points_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {points_path}")

    # ESF curves for a handful of samples
    esf_path = out_dir / f"{image_path.stem}_esf.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in samples[:args.n_plot]:
        width = esf_width_10_90(s["s"], s["esf"])
        ax.plot(s["s"], s["esf"], marker=".", markersize=2, linewidth=0.8,
                label=f"row{s['row']} col{s['col']} {s['edge']} d={s['depth']:.2f}m "
                      f"w10-90={width:.2f}px")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("distance along scan line, centered on edge (px)")
    ax.set_ylabel("intensity")
    ax.set_title(f"ESF samples: {image_path.name}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(esf_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {esf_path}")


if __name__ == "__main__":
    main()
