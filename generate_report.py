"""Buat hasil per citra (format: area bertanda tangan vs area kosong):
 - gambar per citra/area  -> outputs/per_image/<citra>_area_<area>_threshold_morphology.png
 - bagian "Hasil per citra" -> ditulis ke README.md (di antara penanda, aman dijalankan ulang)
Jalankan:  python generate_report.py
"""
import glob, os, re, sys
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sigdetect import (ROIS_POSITIVE, ROIS_NEGATIVE, crop_roi, to_gray, thresholds, features, decide)
import cv2 as _cv

OUT = "outputs/per_image"
AREAS = [  # (kunci file, judul, ROI, ada tanda tangan?)
    ("rektor", "Area Rektor (Ada TTD)", ROIS_POSITIVE["rektor"], True),
    ("dekan", "Area Dekan (Ada TTD)", ROIS_POSITIVE["dekan"], True),
    ("blank", "Area Blank (Tanpa TTD)", ROIS_NEGATIVE["kosong_kiri"], False),
]
START, END = "<!-- HASIL-PER-CITRA:START -->", "<!-- HASIL-PER-CITRA:END -->"


def kern(k):
    return _cv.getStructuringElement(_cv.MORPH_ELLIPSE, (k, k))


def analyse(img, roi):
    crop = crop_roi(img, roi)
    gray = to_gray(crop)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th, t = thresholds(gray)
    raw = th["otsu"]
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kern(2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kern(5))
    r = dict(crop=crop, gray=gray, th=th, t=t, opened=opened, closed=closed, total=gray.size)
    r["f_global"] = features(th["global"], blur)
    r["f_otsu_raw"] = features(raw, blur)
    r["f_adapt_raw"] = features(th["adaptive"], blur)
    r["f_open"] = features(opened, blur)
    r["f_final"] = features(closed, blur)
    r["status"] = decide(r["f_final"])[10:]
    return r


def make_figure(name, key, title, r):
    fig, ax = plt.subplots(2, 4, figsize=(11, 4.6))
    items = [
        (cv2.cvtColor(r["crop"], cv2.COLOR_BGR2RGB), "ROI (crop)"),
        (r["gray"], "Grayscale"),
        (r["th"]["global"], f"Global T=127\nfg={r['f_global'].fg_pixels}px"),
        (r["th"]["otsu"], f"Otsu T={r['t']:.0f}\nfg={r['f_otsu_raw'].fg_pixels}px"),
        (r["th"]["adaptive"], f"Adaptive\nfg={r['f_adapt_raw'].fg_pixels}px"),
        (r["opened"], f"Otsu + Opening\nfg={r['f_open'].fg_pixels}px, komp={r['f_open'].n_components}"),
        (r["closed"], f"Opening + Closing (final)\nfg={r['f_final'].fg_pixels}px, komp={r['f_final'].n_components}"),
        (None, ""),
    ]
    for a, (im, ti) in zip(ax.ravel(), items):
        a.axis("off")
        if im is None:
            f = r["f_final"]
            a.text(0.0, 0.5, f"{title}\n\nStatus: SIGNATURE {r['status']}\n"
                   f"Foreground: {f.fg_pixels} / {r['total']}\nKepadatan: {f.fg_ratio:.4f}\n"
                   f"Komponen: {f.n_components}", fontsize=9, va="center")
            continue
        a.imshow(im, cmap=None if im.ndim == 3 else "gray"); a.set_title(ti, fontsize=8)
    plt.suptitle(f"{name} - {title}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}_area_{key}_threshold_morphology.png", dpi=110)
    plt.close()


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def analysis_signed(r):
    g, o, a, op, fi = r["f_global"], r["f_otsu_raw"], r["f_adapt_raw"], r["f_open"], r["f_final"]
    rel = pct(g.fg_pixels, o.fg_pixels)
    if rel < 40:
        gl = (f"Global (T=127) hanya menangkap {g.fg_pixels} piksel ({rel:.0f}% dari hasil Otsu): goresan tipis "
              f"terputus atau hilang karena intensitasnya di atas 127 (false negative).")
    elif rel < 85:
        gl = f"Global (T=127) menangkap {g.fg_pixels} piksel ({rel:.0f}% dari Otsu): sebagian goresan hilang."
    else:
        gl = f"Global (T=127) menangkap {g.fg_pixels} piksel ({rel:.0f}% dari Otsu): hasil mendekati Otsu."
    ot = f"Otsu memilih T={r['t']:.0f} secara otomatis dan menangkap {o.fg_pixels} piksel foreground ({o.n_components} komponen)."
    ad = (f"Adaptive menangkap {a.fg_pixels} piksel dengan {a.n_components} komponen; "
          + ("terdapat bintik noise tambahan di sekitar goresan." if a.n_components > o.n_components + 1
             else "jumlah komponennya mirip Otsu."))
    mo = (f"Opening: komponen {o.n_components} -> {op.n_components}, foreground {o.fg_pixels} -> {op.fg_pixels} piksel. "
          f"Closing: komponen {op.n_components} -> {fi.n_components}, foreground {op.fg_pixels} -> {fi.fg_pixels} piksel. "
          f"Hasil akhir: komponen terbesar mencakup {fi.connectivity:.0%} dari seluruh foreground (connectivity).")
    return gl, ot, ad, mo


def analysis_blank(r):
    g, o, a, op, fi = r["f_global"], r["f_otsu_raw"], r["f_adapt_raw"], r["f_open"], r["f_final"]
    gl = f"Global (T=127): {g.fg_pixels} piksel foreground; kertas kosong lebih terang dari 127 sehingga hampir bersih."
    if o.fg_ratio > 0.10:
        ot = (f"Otsu (T={r['t']:.0f}) menghasilkan {o.fg_pixels} piksel foreground ({o.fg_ratio:.1%} dari ROI). "
              f"Ini bukan tinta: Otsu selalu membagi histogram menjadi dua kelas, sehingga pada ROI tanpa tinta "
              f"ia memotong gradien/noise kertas.")
    else:
        ot = f"Otsu (T={r['t']:.0f}) menghasilkan {o.fg_pixels} piksel foreground ({o.fg_ratio:.1%} dari ROI), hanya bintik kecil."
    ad = f"Adaptive menghasilkan {a.fg_pixels} piksel foreground ({a.n_components} komponen)."
    mo = (f"Morfologi: komponen {o.n_components} -> {fi.n_components}, foreground {o.fg_pixels} -> {fi.fg_pixels} piksel. "
          f"Keputusan ABSENT ditentukan oleh syarat kontras tinta: ink_contrast = {fi.ink_contrast:.3f} (< 0,10), "
          f"artinya tidak ada piksel yang benar-benar gelap dibanding kertas.")
    return gl, ot, ad, mo


def section(idx, name, label, per_area):
    md = [f"### {idx}. Citra {idx} ({label})\n"]
    for key, title, truth, r in per_area:
        f = r["f_final"]
        md.append(f"**{title}**\n")
        md.append(f"| Karakteristik (Otsu + opening + closing) | Nilai |\n|---|---|\n"
                  f"| Status | **SIGNATURE {r['status']}** |\n"
                  f"| Foreground pixels | {f.fg_pixels} / {r['total']} |\n"
                  f"| Kepadatan piksel | {f.fg_ratio:.4f} |\n"
                  f"| Komponen terhubung | {f.n_components} |\n"
                  f"| Connectivity | {f.connectivity:.2f} |\n"
                  f"| Kontras tinta | {f.ink_contrast:.3f} |\n")
        gl, ot, ad, mo = (analysis_signed if truth else analysis_blank)(r)
        md.append(f"**Analisis perbandingan**\n\n- Thresholding\n  - {gl}\n  - {ot}\n  - {ad}\n- Morfologi\n  - {mo}\n")
        md.append(f"![{name} {key}](outputs/per_image/{name}_area_{key}_threshold_morphology.png)\n")
        benar = (r["status"] == "PRESENT") == truth
        md.append(f"Hasil sesuai harapan: **{'Ya' if benar else 'TIDAK'}**\n")
    return "\n".join(md)


def main():
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob("data/certs/*.png"))
    parts, total, benar = [], 0, 0
    for i, p in enumerate(paths, 1):
        name = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        per_area = []
        for key, title, roi, truth in AREAS:
            r = analyse(img, roi)
            make_figure(name, key, title, r)
            per_area.append((key, title, truth, r))
            total += 1; benar += (r["status"] == "PRESENT") == truth
        parts.append(section(i, name, name[3:], per_area))
    body = (f"{START}\n## Hasil per citra\n\n"
            f"Ringkasan: **{benar}/{total}** area diklasifikasikan benar "
            f"(2 area bertanda tangan + 1 area kosong, pada 9 varian citra).\n\n"
            + "\n".join(parts) + f"\n{END}\n")
    readme = open("README.md", encoding="utf-8").read()
    if START in readme:
        readme = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n?", lambda m: body, readme, flags=re.S)
    else:
        readme = readme.rstrip() + "\n\n" + body
    open("README.md", "w", encoding="utf-8").write(readme)
    print(f"Selesai: {benar}/{total} benar. Gambar di {OUT}/, README.md diperbarui.")


if __name__ == "__main__":
    main()
