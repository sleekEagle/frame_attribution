"""
def_calibration/build_writeup.py -- one-off script generating a PDF writeup of the D_focus
calibration methodology (theory + method), pulling the results table straight from
dfocus_results.txt. Not part of the calibration pipeline itself; run once by hand.

    python def_calibration/build_writeup.py
"""
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 PageBreak)

RESULTS_CSV = Path(r"D:\datasets\MODEST_processed\scene4_dfocus\dfocus_results.txt")
OUT_PDF = Path(r"D:\datasets\MODEST_processed\scene4_dfocus\focaldist_estimation_writeup.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("Eq", parent=styles["Normal"], fontName="Courier",
                           fontSize=10, leftIndent=24, spaceBefore=6, spaceAfter=6))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], spaceBefore=4, spaceAfter=8,
                           leading=15))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6))


def p(text, style="Body"):
    return Paragraph(text, styles[style])


def eq(text):
    return Paragraph(text, styles["Eq"])


def build_story():
    story = []

    story.append(Paragraph("Focus-Distance (D<sub>focus</sub>) Calibration from Paired "
                            "Color/Depth Images", styles["Title"]))
    story.append(p("Methodology writeup for def_calibration/focaldist_estimation.py, "
                    "applied to the MODEST scene4 dataset.", "Italic"))
    story.append(Spacer(1, 12))

    # ---- 1. Overview ------------------------------------------------------------------
    story.append(p("1. Overview", "H1"))
    story.append(p(
        "For each camera setting (focal length &times; left/right camera), this pipeline "
        "estimates the physical distance the lens was focused at (D<sub>focus</sub>) purely "
        "from a set of ordinary photographs and their paired ground-truth depth maps -- no "
        "additional calibration target or hardware readout is needed. The approach measures "
        "how image sharpness varies with scene depth in each photo, and fits that "
        "relationship to the physical model of defocus blur to recover the depth at which "
        "blur is minimized."))

    # ---- 2. Theory ----------------------------------------------------------------------
    story.append(p("2. Theory", "H1"))

    story.append(p("2.1 Thin-lens equation and the sensor plane", "H2"))
    story.append(p(
        "The thin-lens equation relates an object distance u, the resulting image distance "
        "v (where light from that object converges to a sharp point behind the lens), and "
        "the lens's focal length f:"))
    story.append(eq("1/f = 1/v + 1/u"))
    story.append(p(
        "A camera focused at distance D<sub>focus</sub> has its sensor fixed at the image "
        "distance v<sub>s</sub> that brings objects at exactly D<sub>focus</sub> into sharp "
        "focus:"))
    story.append(eq("v<sub>s</sub> = f &middot; D<sub>focus</sub> / (D<sub>focus</sub> - f)"))

    story.append(p("2.2 Defocus blur (circle of confusion)", "H2"))
    story.append(p(
        "An object actually located at some other true depth d would converge to a sharp "
        "point at a different image distance v<sub>d</sub> = f&middot;d / (d - f). Since the "
        "sensor sits at v<sub>s</sub> instead, that object's image is a blur disc rather "
        "than a point -- the classic 'circle of confusion'. By similar triangles, its "
        "diameter scales with the aperture diameter A and how far v<sub>s</sub> is from the "
        "true convergence point v<sub>d</sub>:"))

    story.append(eq("c(d) = A &middot; |v<sub>s</sub> - v<sub>d</sub>| / v<sub>d</sub>"))
    story.append(p(
        "Substituting v<sub>s</sub> and v<sub>d</sub> and simplifying algebraically "
        "(the f and d/D<sub>focus</sub> terms combine and mostly cancel) gives a clean "
        "result:"))
    story.append(eq("c(d) = k &middot; |1/d - 1/D<sub>focus</sub>|,  "
                     "where  k = A&middot;f&middot;D<sub>focus</sub> / (D<sub>focus</sub> - f)"))
    story.append(p(
        "This is the central physical fact the whole calibration rests on: blur grows "
        "linearly with how far 1/d is from 1/D<sub>focus</sub>, hitting exactly zero at "
        "d = D<sub>focus</sub>."))

    story.append(p("2.3 Aperture and f-number", "H2"))
    story.append(p(
        "The f-number N (e.g. F2.8, F16) is defined as N = f / A, i.e. A = f / N. "
        "Substituting into k above shows that k is inversely proportional to the f-number "
        "for a fixed focal length and D<sub>focus</sub>:"))
    story.append(eq("k<sub>i</sub> = K<sub>0</sub> / N<sub>i</sub>,  "
                     "where  K<sub>0</sub> = f&sup2;&middot;D<sub>focus</sub> / "
                     "(D<sub>focus</sub> - f)"))
    story.append(p(
        "K<sub>0</sub> is a single constant shared by every f-stop of the same "
        "(focal length, side) setting: aperture only changes how much blur grows away from "
        "focus, not where the focus point is. This relationship is what lets the different "
        "f-stops captured for one camera setting be combined into a single, properly "
        "constrained fit (Section 3.3) rather than treated as unrelated data."))

    story.append(p("2.4 Laplacian variance as a practical blur proxy", "H2"))
    story.append(p(
        "c(d) itself is not directly observable from a photograph. Instead, this pipeline "
        "measures local image sharpness with the variance of the Laplacian filter response "
        "within small patches -- a standard, cheap autofocus/sharpness metric (a single "
        "small convolution plus a windowed variance, fully vectorizable). Laplacian "
        "variance is <i>not</i> linearly proportional to the true blur diameter c the way, "
        "e.g., an edge-spread-width measurement would be: it is a sharpness metric that "
        "responds most strongly to a scene's own high-frequency (fine-texture) content, and "
        "its exact quantitative relationship to c depends on that scene's own texture power "
        "spectrum -- not something derivable in closed form for an arbitrary scene."))
    story.append(p(
        "A frequency-domain argument (via Parseval's theorem, weighting a scene's power "
        "spectrum by both the Laplacian's f&sup2; response and the defocus PSF's own "
        "frequency attenuation) shows Laplacian variance falls off with blur width "
        "approximately as a power law, with an exponent that itself depends on the scene's "
        "texture spectrum -- e.g. roughly &sigma;&#8315;&#8309; for a flat/white spectrum "
        "versus roughly &sigma;&#8315;&#179; for the 1/f&sup2; spectrum typical of natural "
        "images. Rather than assume either, this pipeline leaves that exponent as a free "
        "fit parameter (p below), letting the data determine it per camera setting. "
        "Laplacian variance is also a <i>sharpness</i> metric -- maximized at "
        "D<sub>focus</sub>, not zero there like c(d) itself -- which the fitted model "
        "accommodates by allowing p to be negative."))

    story.append(p("2.5 Combined fitting model", "H2"))
    story.append(p(
        "Combining the aperture relationship (2.3) with the free-exponent power law (2.4) "
        "gives the model actually fit to the data, per f-stop i:"))
    story.append(eq("blur_metric<sub>i</sub>(d) &asymp; K &middot; N<sub>i</sub>"
                     "<super>-p</super> &middot; "
                     "(|1/d - 1/D<sub>focus</sub>| + &epsilon;)<super>p</super>"))
    story.append(p(
        "with K, D<sub>focus</sub>, and p all free but <i>shared</i> across every f-stop of "
        "a given (focal length, side) setting -- only the known N<sub>i</sub> differs per "
        "f-stop. &epsilon; is a small fixed numerical floor (in depth-normalized units) that "
        "keeps the model finite exactly at d = D<sub>focus</sub> instead of singular, since "
        "a literal zero-distance floor would blow up for negative p."))

    story.append(PageBreak())

    # ---- 3. Method ------------------------------------------------------------------
    story.append(p("3. Method", "H1"))

    story.append(p("3.1 Data layout and patch extraction", "H2"))
    story.append(p(
        "For each (focal length, L/R side) camera setting, every f-stop's color/depth image "
        "pairs are pooled. Each image is split into non-overlapping 32&times;32 patches. "
        "A patch contributes a (depth, blur) sample only if it passes three checks: "
        "(1) at least 90% of its pixels have finite, positive depth; (2) its depth "
        "coefficient of variation (std/mean) is below 5%, i.e. it does not straddle a depth "
        "discontinuity such as an object edge; (3) its grayscale pixel standard deviation "
        "exceeds a minimum threshold, i.e. it has enough texture for a sharpness measurement "
        "to be meaningful at all. The blur metric itself is the variance of "
        "scipy.ndimage.laplace() applied to the patch; the paired depth is the median depth "
        "of its valid pixels."))

    story.append(p("3.2 Depth binning with a high percentile", "H2"))
    story.append(p(
        "Laplacian variance depends heavily on how much texture a patch happens to contain, "
        "not just on blur -- a weakly-textured patch reads as \u201cblurry\u201d even in "
        "perfect focus, simply because it has little high-frequency content to lose. This is "
        "the dominant source of scatter in the raw (depth, blur) samples, and it is a bias "
        "per patch rather than estimation noise, so neither more samples nor larger patches "
        "removes it. Instead, each f-stop's samples are divided into 40 bins across its "
        "observed depth range, and the 90th percentile of blur within each bin (rather than "
        "the mean or the raw samples) is taken as that bin's representative point. This "
        "favors the best-textured patches at each depth, which most faithfully reveal the "
        "true blur level there, largely cancelling the texture-content bias rather than "
        "averaging over it. Binning is done separately per f-stop, before combining, since "
        "mixing different f-stops' raw samples into one bin would let the largest-aperture "
        "f-stop's larger blur values dominate the percentile."))

    story.append(p("3.3 Joint multi-f-stop curve fit", "H2"))
    story.append(p(
        "All f-stops' binned points for a given (focal length, side) setting are pooled, "
        "each carrying its own known f-number, and fit jointly via nonlinear least squares "
        "(scipy.optimize.curve_fit) to the model in Section 2.5. Depth and f-number are "
        "normalized (divided by their medians) before fitting, keeping the optimizer's "
        "parameters near order-1 scale for numerical stability; results are rescaled back "
        "afterward. The initial guess for D<sub>focus</sub> is the depth of the single "
        "sharpest observed sample; bounds constrain the fitted D<sub>focus</sub> to a "
        "plausible range around the observed depth span, and the exponent p to "
        "[-15, 15] as a loose numerical safeguard rather than a tight physical constraint. "
        "This joint approach -- one shared D<sub>focus</sub> and K, with each f-stop's "
        "contribution scaled by its own known N<sub>i</sub> -- is what actually ties the "
        "f-stops together correctly: fitting each f-stop independently discards the known "
        "aperture relationship, while naively pooling all f-stops' raw samples under one "
        "shared blur scale (ignoring that scale genuinely differs by aperture) proved "
        "unstable in practice, producing implausible outlier estimates for some settings."))

    story.append(p("3.4 Outputs", "H2"))
    story.append(p(
        "For each setting, the fit returns: the estimated D<sub>focus</sub>; its parameter "
        "standard error from the fit's covariance matrix (d_focus_stderr -- how confidently "
        "constrained that one number is); the fitted K and p; and the fit's overall RMSE "
        "against the binned points (a broader fit-quality measure). A diagnostic plot per "
        "setting shows the raw patch samples, the binned points used for fitting (colored "
        "per f-stop), and each f-stop's fitted curve overlaid, with a vertical line marking "
        "the fitted D<sub>focus</sub>."))

    story.append(PageBreak())

    # ---- 4. Results ------------------------------------------------------------------
    story.append(p("4. Results", "H1"))
    story.append(p(
        "Fitted for all 10 focal lengths &times; 2 sides (20 camera settings total) on the "
        "MODEST scene4 dataset. Depth units match the input depth TIFFs (the fit itself is "
        "unit-agnostic)."))
    story.append(Spacer(1, 6))

    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        header = ["focal", "side", "n_fstops", "n_images", "d_focus", "d_focus_stderr", "p",
                  "rmse"]
        table_data = [header]
        for r in rows:
            table_data.append([
                r["focal"], r["side"], r["n_fstops"], r["n_images"],
                f"{float(r['d_focus']):.4f}" if r["d_focus"] else "-",
                f"{float(r['d_focus_stderr']):.5f}" if r["d_focus_stderr"] else "-",
                f"{float(r['p']):.3f}" if r["p"] else "-",
                f"{float(r['rmse']):.2f}" if r["rmse"] else "-",
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

    story.append(Spacer(1, 14))
    story.append(p(
        "L/R agreement within each focal length is a useful sanity check on the fit: the "
        "two cameras' D<sub>focus</sub> values should be close if they share the same focus "
        "setting for a given focal length, which is what the fitted values above show."))

    # ---- 5. References ------------------------------------------------------------------
    story.append(p("5. Code reference", "H1"))
    story.append(p(
        "Implementation: def_calibration/focaldist_estimation.py. "
        "Results: dfocus_results.txt. Diagnostic plots: plots/&lt;focal&gt;_&lt;side&gt;.png, "
        "all in D:\\datasets\\MODEST_processed\\scene4_dfocus\\."))

    return story


def main():
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                             leftMargin=0.85 * inch, rightMargin=0.85 * inch)
    doc.build(build_story())
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
