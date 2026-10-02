"""Modul inti: crop ROI -> grayscale -> thresholding -> morfologi -> fitur -> aturan keputusan."""
from dataclasses import dataclass
import cv2
import numpy as np

# ROI relatif (x1, y1, x2, y2) terhadap ukuran citra ijazah (sudah ditegakkan/rotasi).
# Relatif -> tetap valid walau resolusi tiap varian citra berbeda.
ROIS_POSITIVE = {
    "rektor":  (0.13, 0.70, 0.41, 0.83),   # tanda tangan Rektor (kepala institusi)
    "dekan":   (0.63, 0.69, 0.88, 0.83),   # tanda tangan Dekan
    "pemilik": (0.43, 0.855, 0.52, 0.925),  # tanda tangan pemilik ijazah
}
ROIS_NEGATIVE = {
    "kosong_kiri":  (0.02, 0.70, 0.12, 0.83),  # area kertas kosong
    "kosong_atas":  (0.05, 0.03, 0.30, 0.15),  # area kertas kosong
    "teks_nama":    (0.13, 0.835, 0.41, 0.88),  # teks tercetak (hard negative)
    "teks_nomor":   (0.05, 0.90, 0.38, 0.96),   # teks tercetak (hard negative)
}
ROI_WIDTH = 300  # semua ROI di-resize ke lebar ini agar ukuran piksel & kernel konsisten


def crop_roi(img, roi, width=ROI_WIDTH):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    c = img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    s = width / c.shape[1]
    return cv2.resize(c, (width, max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_CUBIC)


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def thresholds(gray, global_t=127, block=31, c=12):
    """Mengembalikan dict {nama: biner}. Foreground (tinta) = 255."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    out = {}
    _, out["global"] = cv2.threshold(blur, global_t, 255, cv2.THRESH_BINARY_INV)
    t_otsu, out["otsu"] = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    out["adaptive"] = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, block, c)
    return out, t_otsu


def morphology(binary, open_k=2, close_k=5):
    """Opening (hapus bintik noise) lalu closing (sambung goresan putus)."""
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ko)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kc)


@dataclass
class Features:
    fg_pixels: int
    fg_ratio: float
    n_components: int
    largest_cc: int
    largest_cc_ratio: float
    bbox_w_ratio: float
    bbox_h_ratio: float
    connectivity: float = 0.0   # largest_cc / fg_pixels: tanda tangan = satu goresan menyambung
    ink_contrast: float = 0.0   # (median latar - persentil-1 piksel tergelap) / median: ada tinta gelap atau tidak


def features(binary, gray=None, min_cc=15):
    h, w = binary.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= min_cc]
    fg = int((binary > 0).sum())
    if gray is not None:
        med = float(np.median(gray))
        contrast = (med - float(np.percentile(gray, 1))) / max(med, 1.0)
    else:
        contrast = 0.0
    if not comps:
        return Features(fg, fg / (h * w), 0, 0, 0.0, 0.0, 0.0, 0.0, contrast)
    largest = max(s[cv2.CC_STAT_AREA] for s in comps)
    x1 = min(s[cv2.CC_STAT_LEFT] for s in comps)
    y1 = min(s[cv2.CC_STAT_TOP] for s in comps)
    x2 = max(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] for s in comps)
    y2 = max(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] for s in comps)
    return Features(fg, fg / (h * w), len(comps), int(largest), largest / (h * w),
                    (x2 - x1) / w, (y2 - y1) / h, largest / max(fg, 1), contrast)


# ---- Aturan keputusan (nilai diset dari eksperimen, lihat README) ----
RULE = dict(min_ink_contrast=0.10,      # ada tinta yang cukup gelap dibanding kertas (cegah false positive area kosong)
            min_fg_ratio=0.03,        # cukup banyak piksel foreground
            max_fg_ratio=0.40,        # terlalu banyak = bukan goresan (mis. tepi kertas/bayangan)
            min_connectivity=0.55,    # goresan dominan harus menyambung (teks tercetak = banyak komponen kecil)
            min_bbox_w=0.35,          # goresan membentang cukup lebar di ROI
            min_bbox_h=0.45)          # dan cukup tinggi (membedakan dari baris teks pendek)


def decide(f: Features, rule=RULE):
    ok = (f.ink_contrast >= rule["min_ink_contrast"]
          and rule["min_fg_ratio"] <= f.fg_ratio <= rule["max_fg_ratio"]
          and f.connectivity >= rule["min_connectivity"]
          and f.bbox_w_ratio >= rule["min_bbox_w"]
          and f.bbox_h_ratio >= rule["min_bbox_h"])
    return "SIGNATURE PRESENT" if ok else "SIGNATURE ABSENT"
