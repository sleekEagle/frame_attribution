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
import json
from pathlib import Path

import cv2
import cv2.aruco as aruco
import matplotlib
matplotlib.use("Agg")
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
                         lines_per_square: int = 3, margin_frac: float = 0.35,
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

    results = []
    for row in range(squares_y):
        for col in range(squares_x):
            for edge, x_edge in (("left", col * sq), ("right", (col + 1) * sq)):
                if (edge == "left" and col == 0) or (edge == "right" and col == squares_x - 1):
                    continue
                for frac in fracs:
                    y_line = (row + frac) * sq
                    obj_pts = np.stack(
                        [x_edge + t, np.full_like(t, y_line), np.zeros_like(t)], axis=1)
                    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
                    img_pts = img_pts.reshape(-1, 2)

                    seg_len = np.linalg.norm(np.diff(img_pts, axis=0), axis=1)
                    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
                    if arc[-1] < 2 * sample_spacing_px:
                        continue
                    s_edge = np.interp(0.0, t, arc)

                    s_uniform = np.arange(0.0, arc[-1], sample_spacing_px)
                    px = np.interp(s_uniform, arc, img_pts[:, 0])
                    py = np.interp(s_uniform, arc, img_pts[:, 1])

                    esf = _bilinear_sample(gray, px, py)
                    if np.isnan(esf).any():
                        continue

                    k = max(3, len(esf) // 10)
                    contrast = abs(float(esf[:k].mean()) - float(esf[-k:].mean()))
                    if contrast < min_contrast:
                        continue

                    obj_edge = np.array([[x_edge, y_line, 0.0]])
                    depth = float((R @ obj_edge.T + tvec)[2, 0])
                    if depth <= 0:
                        continue

                    results.append({
                        "row": row, "col": col, "edge": edge, "frac_height": float(frac),
                        "depth": depth, "s": s_uniform - s_edge, "esf": esf,
                        "contrast": contrast,
                    })
    return results


def main():
    """Run extract_esf_samples() on a single image, for a quick sanity check. The image is
    expected under the usual <calib_root>/<camera_dir>/fl_<X>mm/*.JPG layout (see
    focaldist_estimation_calibimgs.py's module docstring), so calib_root, camera_dir and the
    focal length are inferred from image_path's own parent directories."""
    default_image = (r"C:\Users\lahir\MODEST\Global_calibration_set\MODEST_ChArUco"
                      r"\Global_calibration_set\ChArUco_pattern\EOS_6D_A\fl_28mm\IMG_6495.JPG")
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("image_path", type=Path, nargs="?", default=Path(default_image))
    parser.add_argument("--out_plot", type=Path, default=None,
                         help="where to save example ESF curves (default: next to the image)")
    parser.add_argument("--n_plot", type=int, default=6, help="how many samples to plot")
    args = parser.parse_args()

    image_path = args.image_path
    focal_name = image_path.parent.name  # e.g. "fl_28mm"
    camera_dir = image_path.parent.parent  # e.g. .../EOS_6D_A
    calib_root = camera_dir.parent  # .../ChArUco_pattern

    with open(calib_root / "pattern_info_charuco.json") as f:
        info = json.load(f)["charuco"]
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, info["dictionary"]))
    board = aruco.CharucoBoard((info["squares_x"], info["squares_y"]),
                                info["square_length_m"], info["marker_length_m"], dictionary)
    obj_points_all = board.getChessboardCorners()
    detector = aruco.CharucoDetector(board)

    npz = np.load(camera_dir / "calibration" / f"{focal_name}.npz")
    K, D = npz["K"], npz["D"]

    samples = extract_esf_samples(image_path, detector, obj_points_all, K, D, info)
    print(f"{image_path.name}: {len(samples)} ESF samples")
    if not samples:
        return
    depths = [s["depth"] for s in samples]
    print(f"  depth range: [{min(depths):.3f}, {max(depths):.3f}] m")

    out_plot = args.out_plot or image_path.with_name(image_path.stem + "_esf_samples.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in samples[:args.n_plot]:
        ax.plot(s["s"], s["esf"], marker=".", markersize=2, linewidth=0.8,
                label=f"row{s['row']} col{s['col']} {s['edge']} d={s['depth']:.2f}m")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("distance along scan line, centered on edge (px)")
    ax.set_ylabel("intensity")
    ax.set_title(f"ESF samples: {image_path.name}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_plot}")


if __name__ == "__main__":
    main()
