"""Mini project: deteksi keberadaan tanda tangan pada ijazah.

Contoh:
  python run_pipeline.py                         # batch: semua citra di data/certs + evaluasi + gambar
  python run_pipeline.py --image data/certs/03_blurred.png --roi rektor   # satu citra
"""
import argparse, csv, glob, os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sigdetect import (ROIS_POSITIVE, ROIS_NEGATIVE, crop_roi, to_gray, thresholds,
                       morphology, features, decide)

OUT = "outputs"
METHODS = ["global", "otsu", "adaptive"]


def analyse(img, roi):
    crop = crop_roi(img, roi)
    gray = to_gray(crop)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th, t_otsu = thresholds(gray)
    res = {}
    for m in METHODS:
        clean = morphology(th[m])
        res[m] = dict(binary=th[m], clean=clean, feat=features(clean, blur))
    return crop, gray, res, t_otsu


def build_testset(paths):
    items = []
    for p in paths:
        img = cv2.imread(p)
        name = os.path.splitext(os.path.basename(p))[0]
        for r, roi in ROIS_POSITIVE.items():
            items.append((name, r, roi, "PRESENT", img))
        for r, roi in ROIS_NEGATIVE.items():
            items.append((name, r, roi, "ABSENT", img))
    return items


def run_batch(paths):
    os.makedirs(OUT, exist_ok=True)
    rows, per_method = [], {m: [] for m in METHODS}
    for name, r, roi, truth, img in build_testset(paths):
        crop, gray, res, t = analyse(img, roi)
        row = dict(image=name, roi=r, truth=truth, otsu_T=round(t))
        for m in METHODS:
            f = res[m]["feat"]
            pred = decide(f)[10:]
            per_method[m].append(pred == truth)
            row.update({f"{m}_fg_pixels": f.fg_pixels, f"{m}_fg_ratio": round(f.fg_ratio, 4),
                        f"{m}_pred": pred})
        rows.append(row)
    with open(f"{OUT}/results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    lines = ["| Metode threshold | Akurasi | Benar/Total | FP (negatif→PRESENT) | FN (positif→ABSENT) |", "|---|---|---|---|---|"]
    for m in METHODS:
        fp = sum(1 for rw in rows if rw["truth"] == "ABSENT" and rw[f"{m}_pred"] == "PRESENT")
        fn = sum(1 for rw in rows if rw["truth"] == "PRESENT" and rw[f"{m}_pred"] == "ABSENT")
        ok = sum(per_method[m])
        lines.append(f"| {m} | {ok/len(rows):.1%} | {ok}/{len(rows)} | {fp} | {fn} |")
    with open(f"{OUT}/metrics.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


def fig_threshold_comparison(paths, roi_name="rektor"):
    roi = ROIS_POSITIVE[roi_name]
    cols = ["ROI (crop)", "Grayscale", "Global T=127", "Otsu", "Adaptive Gaussian", "Otsu + Opening/Closing"]
    fig, ax = plt.subplots(len(paths), len(cols), figsize=(2.6 * len(cols), 1.15 * len(paths) + 0.6))
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        crop, gray, res, t = analyse(img, roi)
        ims = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), gray, res["global"]["binary"],
               res["otsu"]["binary"], res["adaptive"]["binary"], res["otsu"]["clean"]]
        for j, im in enumerate(ims):
            ax[i, j].imshow(im, cmap=None if j == 0 else "gray"); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title(cols[j], fontsize=8)
        ax[i, 0].set_ylabel(os.path.basename(p)[:2], rotation=0, labelpad=12, fontsize=8)
    plt.suptitle(f"Perbandingan thresholding - ROI '{roi_name}' pada 9 varian citra", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/threshold_comparison_{roi_name}.png", dpi=130); plt.close()


def fig_morphology(path, roi_name="dekan"):
    img = cv2.imread(path)
    crop, gray, res, t = analyse(img, ROIS_POSITIVE[roi_name])
    raw = res["otsu"]["binary"]
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)); kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, ko); closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kc)
    titles = ["Otsu (mentah)", "Opening 2x2", "Opening + Closing 5x5"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.4))
    for a, im, ti in zip(ax, [raw, opened, closed], titles):
        f = features(im)
        a.imshow(im, cmap="gray"); a.set_title(f"{ti}\nfg={f.fg_pixels}px, komponen={f.n_components}", fontsize=8); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/morphology_effect.png", dpi=130); plt.close()


def fig_threshold_sweep(path, roi_name="rektor"):
    """Efek threshold terlalu rendah / pas / terlalu tinggi."""
    img = cv2.imread(path)
    g = cv2.GaussianBlur(to_gray(crop_roi(img, ROIS_POSITIVE[roi_name])), (3, 3), 0)
    t_otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cases = [("T terlalu RENDAH (T=80)", 80), (f"T pas (Otsu={t_otsu:.0f})", t_otsu), ("T terlalu TINGGI (T=252)", 252)]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.4))
    for a, (ti, T) in zip(ax, cases):
        _, b = cv2.threshold(g, T, 255, cv2.THRESH_BINARY_INV)
        a.imshow(b, cmap="gray"); a.set_title(f"{ti}\nfg_ratio={(b>0).mean():.3f}", fontsize=8); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/threshold_too_low_high.png", dpi=130); plt.close()
    ts = np.arange(0, 256, 5)
    sig = [(g < T).mean() for T in ts]
    blank = cv2.GaussianBlur(to_gray(crop_roi(img, (0.02, 0.70, 0.12, 0.83))), (3, 3), 0)
    bl = [(blank < T).mean() for T in ts]
    plt.figure(figsize=(5.5, 3.2))
    plt.plot(ts, sig, label="ROI tanda tangan"); plt.plot(ts, bl, label="ROI kertas kosong")
    plt.axvline(t_otsu, ls="--", c="gray", label="Otsu"); plt.xlabel("Threshold T"); plt.ylabel("fg_ratio")
    plt.legend(); plt.title("fg_ratio vs threshold"); plt.tight_layout(); plt.savefig(f"{OUT}/fg_ratio_vs_threshold.png", dpi=130); plt.close()


def fig_decisions(paths):
    items = [("rektor", ROIS_POSITIVE["rektor"], "PRESENT"), ("kosong_kiri", ROIS_NEGATIVE["kosong_kiri"], "ABSENT"),
             ("teks_nama", ROIS_NEGATIVE["teks_nama"], "ABSENT")]
    fig, ax = plt.subplots(len(items), 3, figsize=(8, 5.5))
    for i, (n, roi, truth) in enumerate(items):
        img = cv2.imread(paths[2])
        crop, gray, res, t = analyse(img, roi)
        f = res["otsu"]["feat"]
        for j, (im, ti) in enumerate([(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), "ROI"), (res["otsu"]["clean"], "Otsu+morfologi")]):
            ax[i, j].imshow(im, cmap=None if j == 0 else "gray"); ax[i, j].axis("off"); ax[i, j].set_title(ti, fontsize=8)
        ax[i, 2].axis("off")
        ax[i, 2].text(0, 0.5, f"{n}\nfg={f.fg_pixels}px ({f.fg_ratio:.1%})\nconn={f.connectivity:.2f}\n-> {decide(f)}", fontsize=9, va="center")
    plt.tight_layout(); plt.savefig(f"{OUT}/decision_examples.png", dpi=130); plt.close()


def run_single(path, roi_name):
    rois = {**ROIS_POSITIVE, **ROIS_NEGATIVE}
    img = cv2.imread(path)
    if img is None: sys.exit(f"Tidak bisa membaca {path}")
    crop, gray, res, t = analyse(img, rois[roi_name])
    for m in METHODS:
        f = res[m]["feat"]
        print(f"[{m:8s}] fg_pixels={f.fg_pixels:5d} fg_ratio={f.fg_ratio:.3f} komponen={f.n_components} "
              f"connectivity={f.connectivity:.2f} -> {decide(f)}")
    print(f"\nKEPUTUSAN AKHIR (Otsu + morfologi): {decide(res['otsu']['feat'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--roi", default="rektor", choices=list({**ROIS_POSITIVE, **ROIS_NEGATIVE}))
    a = ap.parse_args()
    if a.image:
        run_single(a.image, a.roi)
    else:
        paths = sorted(glob.glob("data/certs/*.png"))
        run_batch(paths)
        fig_threshold_comparison(paths, "rektor"); fig_threshold_comparison(paths, "dekan")
        fig_morphology(paths[2]); fig_threshold_sweep(paths[2]); fig_decisions(paths)
        print(f"\nSelesai. Lihat folder {OUT}/")
