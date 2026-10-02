"""Ekstrak 9 citra ijazah dari PDF, putar tegak, simpan ke data/certs/.
Pemakaian: python extract_from_pdf.py data/IJAZAH_PCD.pdf
Memakai PyMuPDF (pip install pymupdf); jika tidak ada, fallback ke `pdfimages` (poppler-utils)."""
import sys, os, glob, subprocess, tempfile, cv2, numpy as np

NAMES = ["01_highquality_enhanced", "02_low_contrast", "03_blurred", "04_high_noise", "05_lowres_upsampled",
         "06_faded_underexposed", "07_colorshift_warmtint", "08_jpeg_artifacts", "09_combined_degradation"]


def images_pymupdf(path):
    import fitz
    pdf = fitz.open(path)
    for page in pdf:
        buf = np.frombuffer(pdf.extract_image(page.get_images()[0][0])["image"], np.uint8)
        yield cv2.imdecode(buf, cv2.IMREAD_COLOR)


def images_poppler(path):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdfimages", "-png", path, f"{d}/i"], check=True)
        for f in sorted(glob.glob(f"{d}/i-*.png")):
            yield cv2.imread(f)


try:
    import fitz  # noqa
    source = images_pymupdf
except ImportError:
    source = images_poppler

os.makedirs("data/certs", exist_ok=True)
for img, name in zip(source(sys.argv[1]), NAMES):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # citra di PDF tersimpan miring 90 derajat
    cv2.imwrite(f"data/certs/{name}.png", img); print("saved", name, img.shape)
