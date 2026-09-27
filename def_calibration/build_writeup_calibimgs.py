"""
def_calibration/build_writeup_calibimgs.py -- one-off script generating a PDF writeup of the
ChArUco-target-based D_focus calibration method (def_calibration/focaldist_estimation_calibimgs.py),
pulling the results table and two example plots straight from
D:\\datasets\\MODEST_processed\\scene4_dfocus_calibimgs\\. Not part of the calibration pipeline
itself; run once by hand, mirroring def_calibration/build_writeup.py's structure.

    python def_calibration/build_writeup_calibimgs.py
"""
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 PageBreak, Image as RLImage)

OUT_DIR = Path(r"D:\datasets\MODEST_processed\scene4_dfocus_calibimgs")
RESULTS_CSV = OUT_DIR / "dfocus_results.txt"
GOOD_PLOT = OUT_DIR / "plots" / "fl_28mm_L.png"
BAD_PLOT = OUT_DIR / "plots" / "fl_50mm_L.png"
OUT_PDF = OUT_DIR / "focaldist_estimation_calibimgs_writeup.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("Eq", parent=styles["Normal"], fontName="Courier",
                           fontSize=10, leftIndent=24, spaceBefore=6, spaceAfter=6))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], spaceBefore=4, spaceAfter=8,
                           leading=15))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9, leading=12,
                           textColor=colors.HexColor("#555555"), spaceBefore=4,
                           spaceAfter=14))


def p(text, style="Body"):
    return Paragraph(text, styles[style])


def eq(text):
    return Paragraph(text, styles["Eq"])


def fig(path: Path, caption: str, width_in=5.5):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    height_in = width_in * h / w
    return [RLImage(str(path), width=width_in * inch, height=height_in * inch),
            p(caption, "Caption")]


def build_story():
    story = []

    story.append(Paragraph("Focus-Distance (D<sub>focus</sub>) Calibration from ChArUco "
                            "Target Images", styles["Title"]))
    story.append(p("Methodology writeup for "
                    "def_calibration/focaldist_estimation_calibimgs.py, a geometric-target-"
                    "based cross-check against the scene-depth-map-based method in "
                    "def_calibration/focaldist_estimation.py.", "Italic"))
    story.append(Spacer(1, 12))

    # ---- 1. Overview ------------------------------------------------------------------
    story.append(p("1. Overview", "H1"))
    story.append(p(
        "This is a second, independent method for estimating a camera's focus distance "
        "(D<sub>focus</sub>) per focal length, using the same paired dataset's ChArUco "
        "calibration-target photographs rather than natural scene photos with a depth "
        "sensor. Its distinguishing idea: distance to the target is computed exactly and "
        "geometrically, via the board's known physical dimensions and the camera's own "
        "calibrated intrinsics, rather than measured by a separate depth sensor. The "
        "defocus-blur physics and curve-fitting approach are otherwise the same as the "
        "scene-depth-map method (see focaldist_estimation.py's writeup for the full "
        "derivation); this document focuses on what is different here."))

    # ---- 2. Theory ----------------------------------------------------------------------
    story.append(p("2. What is different from the scene-depth-map method", "H1"))

    story.append(p("2.1 Geometric depth via ChArUco pose estimation", "H2"))
    story.append(p(
        "A ChArUco board is a checkerboard overlaid with ArUco markers, whose physical "
        "layout (grid size, square length, marker length) is fully known in advance -- here "
        "from pattern_info_charuco.json (12&#215;16 squares, 4.35&nbsp;cm squares, "
        "3&nbsp;cm markers). For each photograph of the board, cv2.aruco.CharucoDetector "
        "locates the board's corners in the image, and cv2.solvePnP recovers the board's "
        "full 3D pose (rotation and translation) relative to the camera, using that focal "
        "length's calibrated camera matrix and distortion coefficients (from its .npz file, "
        "produced by a separate standard checkerboard calibration process, with sub-pixel "
        "mean reprojection error)."))
    story.append(p(
        "Each detected corner's known 3D position on the board's flat plane is then "
        "transformed into camera coordinates by that pose, and its Z-component read off "
        "directly as its true depth -- exact and geometric, with no depth sensor involved. "
        "A useful side effect: when the board is photographed at an oblique angle (common "
        "in a calibration capture session), different corners of the same photograph sit at "
        "genuinely different depths, so a single image can contribute a spread of "
        "(depth, blur) samples on its own."))

    story.append(p("2.2 A single, fixed aperture", "H2"))
    story.append(p(
        "pattern_info_charuco.json records f_number = 16.0, and the calibration image "
        "folders have no per-f-stop subdivision -- every calibration image at a given focal "
        "length was shot at the same fixed F16 aperture. This removes the need for "
        "focaldist_estimation.py's joint multi-f-stop fit (which ties each f-stop's blur "
        "scale to the others via its known f-number): with only one aperture in play, that "
        "mechanism has nothing to tie together, and the model reduces to its plain, "
        "single-aperture form:"))
    story.append(eq("blur_metric(d) &asymp; k &middot; (|1/d - 1/D<sub>focus</sub>| + "
                     "&epsilon;)<super>p</super>"))
    story.append(p(
        "fit per (focal length, side), pooling every calibration image at that focal "
        "length. F16 is a narrow aperture, meaning a large depth of field and "
        "correspondingly weaker defocus-blur variation across distance than the wider "
        "apertures (down to F2.8) used in the scene-depth-map dataset -- a point that "
        "matters for the reliability discussion in Section 4."))

    story.append(p("2.3 Patch sampling around detected corners", "H2"))
    story.append(p(
        "Rather than a uniform grid of patches, a small 24&#215;24 patch is extracted "
        "around each detected ChArUco corner, and its Laplacian variance is computed the "
        "same way as in the scene-depth-map method. Corners are guaranteed high-contrast, so "
        "unlike natural-scene patches, no depth-flatness check is needed (the depth itself "
        "is exact per point, not averaged over a region that could straddle a discontinuity) "
        "-- only a minimal texture-standard-deviation sanity check is kept as a safeguard."))

    story.append(PageBreak())

    # ---- 3. Method ------------------------------------------------------------------
    story.append(p("3. Method summary", "H1"))
    story.append(p(
        "For each (focal length, side): every calibration image at that focal length is "
        "processed for ChArUco corner detection and PnP pose estimation, giving a "
        "(depth, blur) sample per detected corner. These are pooled, divided into 40 bins "
        "across the observed depth range, and the 90th percentile of blur within each bin "
        "(rather than the mean or raw samples) is taken as that bin's representative point "
        "-- the same texture-content-bias fix used in the scene-depth-map method, since a "
        "corner under uneven exposure or near the frame edge can still read as artificially "
        "blurry regardless of true focus. The binned points are then fit via nonlinear least "
        "squares (scipy.optimize.curve_fit) to the model in Section 2.2, with depth "
        "normalized by its median before fitting for numerical stability."))

    # ---- 4. Results ------------------------------------------------------------------
    story.append(p("4. Results", "H1"))
    story.append(p(
        "Fitted for all 10 focal lengths &#215; 2 sides (20 camera settings) using every "
        "available ChArUco calibration image (82-176 images per setting). Depth is in "
        "meters (from the board's physical dimensions in pattern_info_charuco.json)."))
    story.append(Spacer(1, 6))

    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        header = ["focal", "side", "n_images", "n_detected", "d_focus", "d_focus_stderr", "p",
                  "rmse"]
        table_data = [header]
        for r in rows:
            table_data.append([
                r["focal"], r["side"], r["n_images"], r["n_detected"],
                f"{float(r['d_focus']):.3f}" if r["d_focus"] else "-",
                f"{float(r['d_focus_stderr']):.3f}" if r["d_focus_stderr"] else "-",
                f"{float(r['p']):.3f}" if r["p"] else "-",
                f"{float(r['rmse']):.1f}" if r["rmse"] else "-",
            ])
        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ]))
        story.append(tbl)
    else:
        story.append(p(f"[Results file not found at {RESULTS_CSV} at writeup build time.]"))

    story.append(PageBreak())

    # ---- 5. Reliability ------------------------------------------------------------------
    story.append(p("5. Reliability: this method needs per-setting vetting", "H1"))
    story.append(p(
        "Unlike the scene-depth-map method, where left/right camera estimates agreed "
        "closely across every focal length, several settings here show large left/right "
        "disagreement despite deceptively small reported fit uncertainty -- e.g. "
        "fl_40mm (L=1.34m vs R=3.89m) and fl_70mm (L=1.17m vs R=4.05m). Inspecting the "
        "underlying plots explains why."))

    story.append(p("5.1 A reliable fit: fl_28mm, side L", "H2"))
    story.extend(fig(GOOD_PLOT,
                      "Figure 1. A clean, sharp peak in blur vs. distance, closely tracked "
                      "by the fitted curve on both sides -- D_focus = 1.343 m, "
                      "stderr = 0.028 m (2.1% of the estimate). This validates the "
                      "underlying approach when the data actually straddles the true focus "
                      "distance."))

    story.append(p("5.2 An unreliable fit: fl_50mm, side L", "H2"))
    story.extend(fig(BAD_PLOT,
                      "Figure 2. The data forms two disconnected clusters (roughly "
                      "1-1.7 m and 3.7-4.7 m) with a gap between them, and blur is still "
                      "rising at the far edge of the data -- no turnaround is actually "
                      "visible. The fit is extrapolating past the observed range to guess "
                      "where a peak might be, giving D_focus = 9.26 m with "
                      "stderr = 27.6 m: the uncertainty exceeds the estimate itself, i.e. "
                      "this number carries essentially no information."))

    story.append(p("5.3 Cause and implication", "H2"))
    story.append(p(
        "The calibration images appear to have been captured at a handful of discrete "
        "standoff distances rather than a continuous depth sweep, combined with F16's "
        "narrow aperture giving a weaker blur-vs-distance signal than the wider apertures "
        "used in the scene dataset. Together, these can leave a setting's data without a "
        "visible peak, or with only a partial view of one. In that situation the nonlinear "
        "fit can converge to a spurious local optimum that fits its local neighborhood of "
        "bins tightly -- producing a small, reassuring-looking stderr for a value that is "
        "nonetheless wrong. Small stderr alone is therefore not sufficient evidence of a "
        "good fit for this dataset; the settings with the shortest focal lengths "
        "(28-40mm, both sides) show the most plausible, mutually consistent estimates "
        "(roughly 1.1-1.5m) and are the most trustworthy without further checking, while "
        "longer-focal-length settings need per-plot visual verification (or a more robust "
        "multi-start fitting strategy) before being relied on."))

    # ---- 6. References ------------------------------------------------------------------
    story.append(p("6. Code reference", "H1"))
    story.append(p(
        "Implementation: def_calibration/focaldist_estimation_calibimgs.py. "
        "Results: dfocus_results.txt. Diagnostic plots: plots/&lt;focal&gt;_&lt;side&gt;.png, "
        f"all in {OUT_DIR}\\. Calibration images and per-focal-length intrinsics: "
        "C:\\Users\\lahir\\MODEST\\Global_calibration_set\\MODEST_ChArUco\\"
        "Global_calibration_set\\ChArUco_pattern\\."))

    return story


def main():
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                             leftMargin=0.85 * inch, rightMargin=0.85 * inch)
    doc.build(build_story())
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
