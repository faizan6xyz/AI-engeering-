"""
Advanced Image-to-Text Pipeline
=================================
A production-grade pipeline converting images to accurate, structured text.

Pipeline Stages:
  S1.  Input ingestion        — file / URL / base64 / PIL Image
  S2.  Validation & metadata  — format, DPI, EXIF, size guard
  S3.  Image preprocessing    — resize, denoise, contrast, sharpen
  S4.  Binarisation & deskew  — Otsu threshold, Hough-line rotation fix
  S5.  Layout analysis        — region/block detection, reading-order sort
  S6.  OCR (Tesseract)        — per-region parallel extraction
  S7.  Deep-learning OCR      — TrOCR / EasyOCR ensemble (optional)
  S8.  Post-processing        — spell-check, merge regions, clean artefacts
  S9.  Language detection & NLP — langdetect, optional NER
  S10. Output generation      — TXT / JSON / searchable-PDF + quality report

Dependencies:
  pip install pillow pytesseract easyocr transformers torch torchvision
              opencv-python-headless numpy scipy scikit-image
              pyspellchecker langdetect reportlab exifread
  System: sudo apt install tesseract-ocr tesseract-ocr-all poppler-utils
"""

# ────────────── stdlib ──────────────
import io
import os
import re
import json
import math
import time
import logging
import hashlib
import tempfile
import warnings
import urllib.request
import urllib.parse
import base64
from pathlib import Path
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Tuple, Any, Union

# ────────────── third-party ─────────
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ExifTags
from scipy.ndimage import rotate as scipy_rotate
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, disk

warnings.filterwarnings("ignore")

log = logging.getLogger("itt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


# ══════════════════════════════════════════════════════
#  DATA MODELS
# ══════════════════════════════════════════════════════

@dataclass
class ImageMetadata:
    path: str
    width: int
    height: int
    mode: str          # RGB / L / RGBA …
    format: str        # JPEG / PNG / TIFF …
    dpi: Tuple[int, int]
    file_size_kb: float
    checksum: str
    exif: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextRegion:
    """One detected text block / paragraph in the image."""
    index: int
    x: int
    y: int
    w: int
    h: int
    region_type: str        # "text" | "table" | "image" | "header"
    reading_order: int
    raw_text: str = ""
    confidence: float = 0.0
    language: str = "en"


@dataclass
class TranscriptResult:
    """Full result from one image."""
    image_path: str
    regions: List[TextRegion]
    full_text: str
    language: str
    avg_confidence: float
    processing_time_sec: float
    model_used: str


@dataclass
class QualityReport:
    total_chars: int
    total_words: int
    avg_confidence: float
    low_conf_regions: int
    language: str
    processing_time_sec: float
    dpi: Tuple[int, int]
    resolution: str


@dataclass
class PipelineConfig:
    # ── General ──────────────────────────────
    output_formats: List[str] = field(
        default_factory=lambda: ["txt", "json"]
    )
    output_dir: str = "./output"

    # ── Preprocessing ────────────────────────
    max_dimension: int = 4000      # rescale if larger
    min_dimension: int = 300       # upscale if smaller (better OCR)
    target_dpi: int = 300
    denoise_strength: int = 10     # 0 = off, higher = more aggressive
    sharpen: bool = True
    enhance_contrast: bool = True
    contrast_factor: float = 1.5

    # ── Binarisation ─────────────────────────
    binarise: bool = True
    binarise_method: str = "otsu"  # "otsu" | "adaptive" | "sauvola"
    deskew: bool = True
    deskew_max_angle: float = 15.0

    # ── Layout analysis ──────────────────────
    detect_regions: bool = True
    min_region_area: int = 500     # px² — smaller blobs ignored

    # ── OCR engines ──────────────────────────
    tesseract_lang: str = "eng"
    tesseract_psm: int = 6         # 6=uniform block, 3=auto, 11=sparse
    tesseract_oem: int = 3         # 3=LSTM + legacy
    use_easyocr: bool = True
    easyocr_langs: List[str] = field(default_factory=lambda: ["en"])
    use_trocr: bool = False        # heavy GPU model — opt-in
    trocr_model: str = "microsoft/trocr-large-printed"

    # ── Ensemble ─────────────────────────────
    ensemble_strategy: str = "confidence"  # "confidence" | "vote" | "tess_only"

    # ── Post-processing ──────────────────────
    spell_check: bool = True
    spell_check_lang: str = "en"
    min_word_confidence: float = 0.4   # words below this are flagged

    # ── NLP ──────────────────────────────────
    detect_language: bool = True
    extract_entities: bool = False     # requires spacy

    # ── Parallelism ──────────────────────────
    max_workers: int = 4


# ══════════════════════════════════════════════════════
#  S1 — INPUT INGESTION
# ══════════════════════════════════════════════════════
class InputIngester:
    """
    Accepts four source types and returns a PIL Image:
      • Local file path  (str / Path)
      • HTTP/HTTPS URL
      • Base64-encoded string  (data:image/... or raw base64)
      • PIL Image object (pass-through)

    Validates against a whitelist of supported formats before returning.
    """

    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff",
                 ".tif", ".webp", ".gif", ".pbm", ".pgm", ".ppm"}

    def __init__(self, config: PipelineConfig):
        self.config = config

    def ingest(self, source: Union[str, Path, Image.Image]) -> Tuple[Image.Image, str]:
        """Returns (PIL Image, source label string)."""
        log.info("S1 ▶ Ingesting image source …")

        if isinstance(source, Image.Image):
            log.info("  Received PIL Image (%dx%d)", source.width, source.height)
            return source, "pil_image"

        source = str(source)

        if source.startswith("data:image"):
            return self._from_base64(source), "base64"

        if re.match(r"^[A-Za-z0-9+/=\s]{20,}$", source.strip()):
            return self._from_raw_base64(source.strip()), "base64_raw"

        if source.startswith(("http://", "https://")):
            return self._from_url(source), source

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {source}")
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {self.SUPPORTED}"
            )
        img = Image.open(path)
        log.info("  Loaded: %s (%dx%d, mode=%s)", path.name,
                 img.width, img.height, img.mode)
        return img, str(path)

    @staticmethod
    def _from_url(url: str) -> Image.Image:
        log.info("  Downloading %s …", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data))
        log.info("  Downloaded (%dx%d)", img.width, img.height)
        return img

    @staticmethod
    def _from_base64(data_uri: str) -> Image.Image:
        _, encoded = data_uri.split(",", 1)
        raw = base64.b64decode(encoded)
        return Image.open(io.BytesIO(raw))

    @staticmethod
    def _from_raw_base64(b64: str) -> Image.Image:
        raw = base64.b64decode(b64 + "==")     # pad safely
        return Image.open(io.BytesIO(raw))


# ══════════════════════════════════════════════════════
#  S2 — VALIDATION & METADATA EXTRACTION
# ══════════════════════════════════════════════════════
class MetadataExtractor:
    """
    Reads EXIF, DPI, mode, format, and size.
    Applies EXIF orientation correction so downstream stages
    always receive an upright image — a critical step often missed.
    Computes a SHA-256 checksum of the raw pixel bytes.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def extract(self, img: Image.Image, source_label: str) -> Tuple[Image.Image, ImageMetadata]:
        log.info("S2 ▶ Extracting metadata and fixing EXIF orientation …")

        img = self._fix_orientation(img)

        dpi = img.info.get("dpi", (72, 72))
        if isinstance(dpi, (int, float)):
            dpi = (int(dpi), int(dpi))
        else:
            dpi = (int(dpi[0]), int(dpi[1]))

        exif_data = self._read_exif(img)
        checksum  = self._pixel_hash(img)

        size_kb = 0.0
        if Path(source_label).exists():
            size_kb = Path(source_label).stat().st_size / 1024

        meta = ImageMetadata(
            path=source_label,
            width=img.width,
            height=img.height,
            mode=img.mode,
            format=img.format or "unknown",
            dpi=dpi,
            file_size_kb=round(size_kb, 1),
            checksum=checksum,
            exif=exif_data,
        )

        log.info(
            "  %dx%d px | mode=%s | DPI=%s | %.1f KB",
            meta.width, meta.height, meta.mode, meta.dpi, meta.file_size_kb
        )
        return img, meta

    @staticmethod
    def _fix_orientation(img: Image.Image) -> Image.Image:
        """Rotate image according to EXIF orientation tag."""
        try:
            exif = img._getexif()
            if not exif:
                return img
            orient_tag = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            orientation = exif.get(orient_tag, 1)
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                img = img.rotate(rotations[orientation], expand=True)
        except Exception:
            pass
        return img

    @staticmethod
    def _read_exif(img: Image.Image) -> Dict[str, Any]:
        try:
            raw = img._getexif() or {}
            return {
                ExifTags.TAGS.get(k, str(k)): str(v)
                for k, v in raw.items()
                if isinstance(v, (str, int, float, bytes))
            }
        except Exception:
            return {}

    @staticmethod
    def _pixel_hash(img: Image.Image) -> str:
        return hashlib.sha256(img.tobytes()).hexdigest()[:12]


# ══════════════════════════════════════════════════════
#  S3 — IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════
class ImagePreprocessor:
    """
    Applies a chain of classical computer-vision operations to maximise
    OCR accuracy:

    1. Colour normalisation — convert to RGB, remove alpha channel
    2. Resolution guard    — upscale small images, downscale huge ones
    3. Denoising           — OpenCV fastNlMeans (preserves edges better
                             than Gaussian blur)
    4. Contrast enhancement — PIL ImageEnhance.Contrast
    5. Sharpening           — PIL UnsharpMask (radius=2, percent=150)

    Returns a PIL Image ready for binarisation.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def process(self, img: Image.Image) -> Image.Image:
        log.info("S3 ▶ Preprocessing image …")

        # 1. Ensure RGB (drop alpha, convert grayscale)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 2. Resolution guard
        img = self._scale(img)

        # 3. Denoise (operate in numpy/OpenCV space)
        if self.config.denoise_strength > 0:
            arr = np.array(img)
            h   = self.config.denoise_strength
            arr = cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)
            img = Image.fromarray(arr)
            log.info("  Denoised (h=%d).", h)

        # 4. Contrast
        if self.config.enhance_contrast:
            img = ImageEnhance.Contrast(img).enhance(self.config.contrast_factor)
            log.info("  Contrast ×%.1f applied.", self.config.contrast_factor)

        # 5. Sharpen
        if self.config.sharpen:
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            log.info("  Sharpened.")

        log.info("  Preprocessed image: %dx%d", img.width, img.height)
        return img

    def _scale(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_d = self.config.max_dimension
        min_d = self.config.min_dimension

        # Upscale if too small
        if max(w, h) < min_d:
            scale = min_d / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            log.info("  Upscaled to %dx%d (×%.1f).", new_w, new_h, scale)

        # Downscale if too large
        elif max(w, h) > max_d:
            scale = max_d / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            log.info("  Downscaled to %dx%d (×%.2f).", new_w, new_h, scale)

        return img


# ══════════════════════════════════════════════════════
#  S4 — BINARISATION & DESKEW
# ══════════════════════════════════════════════════════
class BinariserDeskewer:
    """
    Two distinct operations:

    Binarisation — converts the preprocessed colour image to strict
    black-and-white.  Three methods are supported:
      • otsu      — global optimal threshold (fast, good for clean docs)
      • adaptive  — local thresholding (handles uneven lighting)
      • sauvola   — local + contrast-aware (best for degraded docs)

    Deskew — detects and corrects page tilt using the Hough line
    transform on the binarised image.  Only corrects angles within
    `deskew_max_angle` degrees to avoid false-positive rotations.
    Returns a PIL Image in grayscale (L mode).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def process(self, img: Image.Image) -> Image.Image:
        log.info("S4 ▶ Binarising and deskewing …")

        gray = np.array(img.convert("L"))

        if self.config.binarise:
            gray = self._binarise(gray)
            log.info("  Binarised using '%s'.", self.config.binarise_method)

        if self.config.deskew:
            angle = self._detect_skew(gray)
            if abs(angle) > 0.3:
                gray = scipy_rotate(gray, angle, reshape=False, cval=255)
                log.info("  Deskewed by %.2f°.", angle)
            else:
                log.info("  No significant skew (%.2f°).", angle)

        return Image.fromarray(gray.astype(np.uint8))

    def _binarise(self, gray: np.ndarray) -> np.ndarray:
        method = self.config.binarise_method

        if method == "otsu":
            thresh = threshold_otsu(gray)
            binary = (gray > thresh).astype(np.uint8) * 255

        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 10
            )

        elif method == "sauvola":
            from skimage.filters import threshold_sauvola
            thresh = threshold_sauvola(gray, window_size=25)
            binary = (gray > thresh).astype(np.uint8) * 255

        else:
            raise ValueError(f"Unknown binarise method: {method}")

        # Morphological opening — remove tiny noise blobs
        cleaned = binary_opening(binary > 128, disk(1))
        return (cleaned * 255).astype(np.uint8)

    def _detect_skew(self, gray: np.ndarray) -> float:
        """
        Detect skew angle via Hough line transform.
        Returns the median angle of detected lines (degrees, CCW positive).
        """
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=10
        )
        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < self.config.deskew_max_angle:
                angles.append(angle)

        return -float(np.median(angles)) if angles else 0.0


# ══════════════════════════════════════════════════════
#  S5 — LAYOUT ANALYSIS
# ══════════════════════════════════════════════════════
class LayoutAnalyser:
    """
    Segments the binarised image into meaningful text regions using
    morphological connected-components analysis:

    1. Dilate horizontally and vertically to merge nearby characters
       into word/line blocks.
    2. Find external contours of the merged blobs.
    3. Filter by minimum area and aspect ratio.
    4. Classify region type heuristically:
         - Very wide + near top → "header"
         - Large area with near-square aspect → "table" (heuristic)
         - Otherwise → "text"
    5. Sort by reading order (top-to-bottom, left-to-right per line).

    Returns a list of TextRegion objects with bounding boxes.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def analyse(self, img: Image.Image) -> List[TextRegion]:
        log.info("S5 ▶ Detecting text regions …")

        gray = np.array(img)
        # Invert so text = white (required for morphological ops)
        binary = cv2.bitwise_not(gray)

        # Dilate to merge characters into blocks
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        dilated  = cv2.dilate(binary, h_kernel, iterations=3)
        dilated  = cv2.dilate(dilated, v_kernel, iterations=3)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: List[TextRegion] = []
        img_h, img_w = gray.shape

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < self.config.min_region_area:
                continue

            rtype = self._classify_region(x, y, w, h, img_w, img_h)

            regions.append(TextRegion(
                index=i,
                x=x, y=y, w=w, h=h,
                region_type=rtype,
                reading_order=0,    # assigned below
            ))

        regions = self._sort_reading_order(regions, img_w)
        for order, r in enumerate(regions):
            r.reading_order = order

        log.info("  Found %d text region(s).", len(regions))
        return regions

    @staticmethod
    def _classify_region(
        x: int, y: int, w: int, h: int, img_w: int, img_h: int
    ) -> str:
        aspect = w / h if h else 1
        if y < img_h * 0.12 and w > img_w * 0.5:
            return "header"
        if 0.7 < aspect < 1.4 and w * h > 40_000:
            return "table"
        return "text"

    @staticmethod
    def _sort_reading_order(
        regions: List[TextRegion], img_w: int
    ) -> List[TextRegion]:
        """
        Group regions into horizontal bands (line-height tolerance),
        then sort left-to-right within each band.
        """
        if not regions:
            return regions

        regions_sorted = sorted(regions, key=lambda r: r.y)
        bands: List[List[TextRegion]] = []
        current_band: List[TextRegion] = [regions_sorted[0]]

        for r in regions_sorted[1:]:
            prev = current_band[-1]
            if r.y < prev.y + prev.h * 0.6:
                current_band.append(r)
            else:
                bands.append(sorted(current_band, key=lambda r: r.x))
                current_band = [r]
        bands.append(sorted(current_band, key=lambda r: r.x))

        return [r for band in bands for r in band]


# ══════════════════════════════════════════════════════
#  S6 — TESSERACT OCR ENGINE
# ══════════════════════════════════════════════════════
class TesseractOCR:
    """
    Runs Google's Tesseract 5 (LSTM engine) on each TextRegion in parallel
    using ThreadPoolExecutor.

    Per-region extraction is more accurate than whole-image OCR because:
      • Each region uses the psm that matches its layout type
        (header → psm 7 single-line, table → psm 6 block, text → config default)
      • Padding is added around each crop to reduce edge-cutting errors

    Confidence scores are extracted from Tesseract's TSV output.
    """

    PSM_MAP = {"header": 7, "table": 6, "text": 6, "image": 11}

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._check_tesseract()

    @staticmethod
    def _check_tesseract():
        import shutil
        if not shutil.which("tesseract"):
            raise RuntimeError(
                "Tesseract not found. Install: sudo apt install tesseract-ocr"
            )

    def ocr_image(
        self, img: Image.Image, regions: List[TextRegion]
    ) -> List[TextRegion]:
        log.info("S6 ▶ Running Tesseract on %d region(s) …", len(regions))

        if not regions:
            # Fallback: run on full image
            return [self._ocr_full(img)]

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {
                pool.submit(self._ocr_region, img, r): r.index
                for r in regions
            }
            for fut in as_completed(futures):
                try:
                    region, text, conf = fut.result()
                    region.raw_text   = text
                    region.confidence = conf
                except Exception as e:
                    log.warning("  Region %d OCR failed: %s", futures[fut], e)

        return regions

    def _ocr_region(
        self, img: Image.Image, region: TextRegion
    ) -> Tuple[TextRegion, str, float]:
        import pytesseract

        pad = 8
        x1 = max(0, region.x - pad)
        y1 = max(0, region.y - pad)
        x2 = min(img.width,  region.x + region.w + pad)
        y2 = min(img.height, region.y + region.h + pad)
        crop = img.crop((x1, y1, x2, y2))

        psm  = self.PSM_MAP.get(region.region_type, self.config.tesseract_psm)
        cfg  = (
            f"--oem {self.config.tesseract_oem} "
            f"--psm {psm} "
            f"-l {self.config.tesseract_lang}"
        )

        data = pytesseract.image_to_data(
            crop, config=cfg, output_type=pytesseract.Output.DICT
        )

        words = [
            w for w, c in zip(data["text"], data["conf"])
            if w.strip() and int(c) != -1
        ]
        confs = [
            int(c) / 100.0
            for w, c in zip(data["text"], data["conf"])
            if w.strip() and int(c) != -1
        ]

        text = " ".join(words)
        conf = float(np.mean(confs)) if confs else 0.0
        return region, text, conf

    def _ocr_full(self, img: Image.Image) -> TextRegion:
        import pytesseract

        cfg  = (
            f"--oem {self.config.tesseract_oem} "
            f"--psm {self.config.tesseract_psm} "
            f"-l {self.config.tesseract_lang}"
        )
        data = pytesseract.image_to_data(
            img, config=cfg, output_type=pytesseract.Output.DICT
        )
        words = [w for w, c in zip(data["text"], data["conf"])
                 if w.strip() and int(c) != -1]
        confs = [int(c) / 100.0 for w, c in zip(data["text"], data["conf"])
                 if w.strip() and int(c) != -1]

        return TextRegion(
            index=0, x=0, y=0, w=img.width, h=img.height,
            region_type="text", reading_order=0,
            raw_text=" ".join(words),
            confidence=float(np.mean(confs)) if confs else 0.0,
        )


# ══════════════════════════════════════════════════════
#  S7 — DEEP-LEARNING OCR ENSEMBLE
# ══════════════════════════════════════════════════════
class DeepLearningOCR:
    """
    Augments Tesseract with one or both deep-learning OCR backends:

    EasyOCR  — CNN + LSTM, good for real-world images and multiple scripts.
               Runs on CPU or GPU automatically.

    TrOCR    — Microsoft's Transformer-based OCR (ViT + GPT-2 decoder).
               Highest accuracy on clean printed text; requires GPU for speed.

    Ensemble strategy:
      "confidence"  — pick text from whichever engine scored higher
      "vote"        — majority-vote on individual words (3+ engines)
      "tess_only"   — skip this stage entirely

    Each region gets the best available text. Tesseract results serve
    as the baseline; deep-learning engines only override when they
    score higher confidence.
    """

    def __init__(self, config: PipelineConfig):
        self.config    = config
        self._easy     = None
        self._trocr_p  = None
        self._trocr_m  = None

        if config.use_easyocr and config.ensemble_strategy != "tess_only":
            self._load_easyocr()

        if config.use_trocr and config.ensemble_strategy != "tess_only":
            self._load_trocr()

    def _load_easyocr(self):
        try:
            import easyocr
            self._easy = easyocr.Reader(
                self.config.easyocr_langs,
                gpu=self._has_gpu(),
                verbose=False,
            )
            log.info("  EasyOCR loaded (langs=%s).", self.config.easyocr_langs)
        except ImportError:
            log.warning("  EasyOCR not installed — skipping.")

    def _load_trocr(self):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            device = "cuda" if self._has_gpu() else "cpu"
            self._trocr_p = TrOCRProcessor.from_pretrained(self.config.trocr_model)
            self._trocr_m = VisionEncoderDecoderModel.from_pretrained(
                self.config.trocr_model
            ).to(device)
            log.info("  TrOCR loaded (%s) on %s.", self.config.trocr_model, device)
        except Exception as e:
            log.warning("  TrOCR not available: %s", e)

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def enhance(
        self, img: Image.Image, regions: List[TextRegion]
    ) -> List[TextRegion]:
        if self.config.ensemble_strategy == "tess_only":
            log.info("S7 ▶ Deep-learning OCR skipped (tess_only).")
            return regions

        log.info("S7 ▶ Running deep-learning OCR ensemble …")

        for region in regions:
            crop = img.crop((region.x, region.y,
                             region.x + region.w, region.y + region.h))

            candidates: List[Tuple[str, float]] = [
                (region.raw_text, region.confidence)
            ]

            # EasyOCR
            if self._easy:
                text, conf = self._run_easyocr(crop)
                if text:
                    candidates.append((text, conf))

            # TrOCR
            if self._trocr_p and self._trocr_m:
                text, conf = self._run_trocr(crop)
                if text:
                    candidates.append((text, conf))

            # Apply ensemble strategy
            if self.config.ensemble_strategy == "confidence":
                best = max(candidates, key=lambda x: x[1])
                region.raw_text   = best[0]
                region.confidence = best[1]
            elif self.config.ensemble_strategy == "vote":
                region.raw_text   = self._vote(candidates)
                region.confidence = max(c for _, c in candidates)

        return regions

    def _run_easyocr(self, crop: Image.Image) -> Tuple[str, float]:
        try:
            results = self._easy.readtext(np.array(crop))
            if not results:
                return "", 0.0
            texts  = [r[1] for r in results]
            confs  = [r[2] for r in results]
            return " ".join(texts), float(np.mean(confs))
        except Exception as e:
            log.debug("  EasyOCR error: %s", e)
            return "", 0.0

    def _run_trocr(self, crop: Image.Image) -> Tuple[str, float]:
        try:
            import torch
            device  = next(self._trocr_m.parameters()).device
            rgb     = crop.convert("RGB")
            inputs  = self._trocr_p(images=rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                ids = self._trocr_m.generate(**inputs)
            text = self._trocr_p.batch_decode(ids, skip_special_tokens=True)[0]
            # TrOCR does not expose per-token confidence easily; proxy = 0.85
            return text.strip(), 0.85
        except Exception as e:
            log.debug("  TrOCR error: %s", e)
            return "", 0.0

    @staticmethod
    def _vote(candidates: List[Tuple[str, float]]) -> str:
        """Word-level majority vote across candidate transcriptions."""
        from collections import Counter

        token_votes: Dict[int, Counter] = {}
        max_len = 0

        for text, _ in candidates:
            words = text.split()
            max_len = max(max_len, len(words))
            for i, w in enumerate(words):
                token_votes.setdefault(i, Counter())[w.lower()] += 1

        return " ".join(
            token_votes[i].most_common(1)[0][0]
            for i in range(max_len)
            if i in token_votes
        )


# ══════════════════════════════════════════════════════
#  S8 — POST-PROCESSING
# ══════════════════════════════════════════════════════
class PostProcessor:
    """
    Cleans and merges raw OCR output:

    1. Artefact removal  — strips common OCR noise: lone punctuation,
       repeated characters ("|||||"), control characters, garbage runs.
    2. Spell correction  — uses pyspellchecker to suggest corrections for
       words below a confidence threshold.  Preserves proper nouns
       (capitalised words) to avoid over-correction.
    3. Region merging    — concatenates regions in reading order, inserting
       paragraph breaks between non-adjacent blocks.
    4. Whitespace normalisation — collapses multiple spaces, trims lines.
    """

    GARBAGE_RE  = re.compile(r"[|\\]{3,}")
    CONTROL_RE  = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
    LONE_PUNCT  = re.compile(r"^\W+$")

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._spell = None

        if config.spell_check:
            try:
                from spellchecker import SpellChecker
                self._spell = SpellChecker(language=config.spell_check_lang)
                log.info("  SpellChecker loaded.")
            except ImportError:
                log.warning("  pyspellchecker not installed — skipping spell-check.")

    def process(self, regions: List[TextRegion]) -> Tuple[List[TextRegion], str]:
        log.info("S8 ▶ Post-processing %d region(s) …", len(regions))

        for r in regions:
            r.raw_text = self._clean(r.raw_text)
            if self._spell and r.confidence < self.config.min_word_confidence + 0.3:
                r.raw_text = self._spell_correct(r.raw_text)

        # Filter empty regions
        regions = [r for r in regions if r.raw_text.strip()]

        full_text = self._merge(regions)
        log.info("  Post-processed: %d chars.", len(full_text))
        return regions, full_text

    def _clean(self, text: str) -> str:
        text = self.GARBAGE_RE.sub("", text)
        text = self.CONTROL_RE.sub("", text)
        lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not self.LONE_PUNCT.match(ln.strip())
        ]
        text = "\n".join(lines)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _spell_correct(self, text: str) -> str:
        words   = text.split()
        unknown = self._spell.unknown(words)
        corrected = []
        for w in words:
            # Preserve capitalised words (likely proper nouns / acronyms)
            if w in unknown and not w[0].isupper():
                suggestion = self._spell.correction(w)
                corrected.append(suggestion or w)
            else:
                corrected.append(w)
        return " ".join(corrected)

    @staticmethod
    def _merge(regions: List[TextRegion]) -> str:
        """Join regions; add blank line between non-adjacent blocks."""
        parts = []
        prev_end_y = -1

        for r in sorted(regions, key=lambda x: x.reading_order):
            gap = (r.y - prev_end_y) if prev_end_y >= 0 else 0
            if gap > r.h * 0.8:
                parts.append("")    # paragraph break
            parts.append(r.raw_text)
            prev_end_y = r.y + r.h

        return "\n".join(parts).strip()


# ══════════════════════════════════════════════════════
#  S9 — LANGUAGE DETECTION & NLP
# ══════════════════════════════════════════════════════
class LanguageNLPProcessor:
    """
    Two optional enrichment steps:

    Language detection — uses langdetect (Naive Bayes classifier trained
    on 55 languages) to identify the script/language of the extracted text.
    Useful for routing to the correct spellcheck dictionary or downstream
    translation.

    Named-entity recognition — uses spaCy (if installed) to extract
    people, organisations, dates, and locations from the transcript.
    Results are embedded in the JSON output as a structured `entities`
    field.  spaCy model must be pre-downloaded:
        python -m spacy download en_core_web_sm
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._nlp   = None

        if config.extract_entities:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                log.info("  spaCy NER loaded.")
            except Exception as e:
                log.warning("  spaCy not available: %s", e)

    def process(self, text: str) -> Tuple[str, List[Dict]]:
        """Returns (detected_language, entities_list)."""
        log.info("S9 ▶ Language detection & NLP …")

        lang = self._detect_language(text)
        entities = self._extract_entities(text) if self._nlp else []

        log.info("  Detected language: %s | Entities: %d", lang, len(entities))
        return lang, entities

    def _detect_language(self, text: str) -> str:
        if not self.config.detect_language or len(text.strip()) < 20:
            return "unknown"
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "unknown"

    def _extract_entities(self, text: str) -> List[Dict]:
        doc = self._nlp(text)
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]


# ══════════════════════════════════════════════════════
#  S10 — OUTPUT GENERATION
# ══════════════════════════════════════════════════════
class OutputGenerator:
    """
    Writes results in all requested formats:

    TXT  — plain UTF-8 text, full_text from merged regions.
    JSON — structured output: metadata, regions, entities, quality report.
    PDF  — searchable PDF (invisible text layer over the original image)
           using reportlab.  This is the "OCR PDF" format used by
           document management systems.

    Also prints and saves a QualityReport JSON.
    """

    def __init__(self, config: PipelineConfig):
        self.config  = config
        self.out_dir = Path(config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        full_text: str,
        regions: List[TextRegion],
        meta: ImageMetadata,
        lang: str,
        entities: List[Dict],
        processing_time: float,
        stem: str,
        original_img: Image.Image,
    ) -> Dict[str, Path]:
        log.info("S10 ▶ Writing output files …")
        outputs: Dict[str, Path] = {}

        avg_conf = (
            float(np.mean([r.confidence for r in regions]))
            if regions else 0.0
        )
        low_conf = sum(1 for r in regions
                       if r.confidence < self.config.min_word_confidence)

        report = QualityReport(
            total_chars=len(full_text),
            total_words=len(full_text.split()),
            avg_confidence=round(avg_conf, 4),
            low_conf_regions=low_conf,
            language=lang,
            processing_time_sec=round(processing_time, 2),
            dpi=meta.dpi,
            resolution=f"{meta.width}x{meta.height}",
        )

        if "txt" in self.config.output_formats:
            outputs["txt"] = self._write_txt(full_text, stem)

        if "json" in self.config.output_formats:
            outputs["json"] = self._write_json(
                full_text, regions, meta, lang, entities, report, stem
            )

        if "pdf" in self.config.output_formats:
            outputs["pdf"] = self._write_searchable_pdf(
                original_img, regions, full_text, stem
            )

        report_path = self.out_dir / f"{stem}_quality.json"
        report_path.write_text(json.dumps(asdict(report), indent=2))
        outputs["quality"] = report_path

        self._print_report(report)
        return outputs

    # ─── Formatters ───────────────────────────────────

    def _write_txt(self, text: str, stem: str) -> Path:
        path = self.out_dir / f"{stem}.txt"
        path.write_text(text, encoding="utf-8")
        log.info("  TXT → %s", path.name)
        return path

    def _write_json(
        self,
        text: str,
        regions: List[TextRegion],
        meta: ImageMetadata,
        lang: str,
        entities: List[Dict],
        report: QualityReport,
        stem: str,
    ) -> Path:
        data = {
            "image": asdict(meta),
            "language": lang,
            "full_text": text,
            "entities": entities,
            "regions": [
                {
                    "index": r.index,
                    "reading_order": r.reading_order,
                    "region_type": r.region_type,
                    "bbox": {"x": r.x, "y": r.y, "w": r.w, "h": r.h},
                    "text": r.raw_text,
                    "confidence": round(r.confidence, 4),
                }
                for r in regions
            ],
            "quality": asdict(report),
        }
        path = self.out_dir / f"{stem}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("  JSON → %s", path.name)
        return path

    def _write_searchable_pdf(
        self,
        img: Image.Image,
        regions: List[TextRegion],
        full_text: str,
        stem: str,
    ) -> Path:
        """
        Build a PDF with the original image as background and invisible
        text positioned over each detected region (hOCR-style PDF).
        Requires reportlab.
        """
        try:
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.units import inch
        except ImportError:
            log.warning("  reportlab not installed — skipping PDF output.")
            return self.out_dir / f"{stem}_NOPDF.txt"

        path    = self.out_dir / f"{stem}.pdf"
        img_w, img_h = img.size

        # Save image to a temp PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        # Create PDF canvas with same pixel dimensions (72 dpi base)
        scale = 72.0 / 96.0   # convert screen pixels to PDF points
        pdf_w = img_w * scale
        pdf_h = img_h * scale

        c = rl_canvas.Canvas(str(path), pagesize=(pdf_w, pdf_h))
        c.drawImage(tmp_path, 0, 0, width=pdf_w, height=pdf_h)

        # Invisible text layer
        c.setFillColorRGB(0, 0, 0, alpha=0)
        c.setFont("Helvetica", 8)

        for r in regions:
            if not r.raw_text.strip():
                continue
            x   = r.x * scale
            y   = pdf_h - (r.y + r.h) * scale     # PDF y is from bottom
            c.drawString(x, y, r.raw_text)

        c.save()
        os.unlink(tmp_path)
        log.info("  Searchable PDF → %s", path.name)
        return path

    # ─── Quality report ───────────────────────────────

    @staticmethod
    def _print_report(r: QualityReport) -> None:
        print("\n" + "═" * 52)
        print("  QUALITY REPORT")
        print("═" * 52)
        print(f"  Characters        : {r.total_chars:,}")
        print(f"  Words             : {r.total_words:,}")
        print(f"  Avg confidence    : {r.avg_confidence:.2%}")
        print(f"  Low-conf regions  : {r.low_conf_regions}")
        print(f"  Language detected : {r.language}")
        print(f"  Input resolution  : {r.resolution} px")
        print(f"  Input DPI         : {r.dpi[0]}×{r.dpi[1]}")
        print(f"  Processing time   : {r.processing_time_sec:.2f} s")
        print("═" * 52 + "\n")


# ══════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════
class ImageToTextPipeline:
    """
    Top-level orchestrator.  Instantiates every stage once, then calls
    `run()` for each image.  Models and engines are reused across calls
    so batch processing is efficient.

    Usage (Python API):
        config   = PipelineConfig(use_easyocr=True, output_formats=["txt","json"])
        pipeline = ImageToTextPipeline(config)
        result   = pipeline.run("invoice.png")
        print(result.full_text)

    Usage (CLI):
        python image_to_text.py invoice.png --model medium --formats txt json pdf
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg        = config or PipelineConfig()
        self.ingester   = InputIngester(self.cfg)
        self.meta_ex    = MetadataExtractor(self.cfg)
        self.preproc    = ImagePreprocessor(self.cfg)
        self.binariser  = BinariserDeskewer(self.cfg)
        self.layout     = LayoutAnalyser(self.cfg)
        self.tess_ocr   = TesseractOCR(self.cfg)
        self.dl_ocr     = DeepLearningOCR(self.cfg)
        self.postproc   = PostProcessor(self.cfg)
        self.nlp        = LanguageNLPProcessor(self.cfg)
        self.output_gen = OutputGenerator(self.cfg)

    def run(
        self,
        source: Union[str, Path, Image.Image],
    ) -> TranscriptResult:
        t0 = time.perf_counter()

        # S1 — Ingest
        img, src_label = self.ingester.ingest(source)
        original_img   = img.copy()

        # S2 — Metadata
        img, meta = self.meta_ex.extract(img, src_label)

        # S3 — Preprocess
        img = self.preproc.process(img)

        # S4 — Binarise & deskew
        bin_img = self.binariser.process(img)

        # S5 — Layout
        regions = self.layout.analyse(bin_img)

        # S6 — Tesseract OCR (on binarised image for accuracy)
        regions = self.tess_ocr.ocr_image(bin_img, regions)

        # S7 — Deep-learning OCR (on colour-preprocessed image for EasyOCR)
        regions = self.dl_ocr.enhance(img, regions)

        # S8 — Post-process
        regions, full_text = self.postproc.process(regions)

        # S9 — Language & NLP
        lang, entities = self.nlp.process(full_text)

        processing_time = time.perf_counter() - t0
        stem            = Path(src_label).stem if src_label != "pil_image" else "output"

        # S10 — Output
        self.output_gen.write(
            full_text, regions, meta, lang, entities,
            processing_time, stem, original_img
        )

        avg_conf = (
            float(np.mean([r.confidence for r in regions])) if regions else 0.0
        )
        model = (
            "trocr" if self.cfg.use_trocr else
            "easyocr+tesseract" if self.cfg.use_easyocr else
            "tesseract"
        )

        return TranscriptResult(
            image_path=src_label,
            regions=regions,
            full_text=full_text,
            language=lang,
            avg_confidence=round(avg_conf, 4),
            processing_time_sec=round(processing_time, 2),
            model_used=model,
        )

    def run_batch(
        self,
        sources: List[Union[str, Path, Image.Image]],
    ) -> List[TranscriptResult]:
        """Process multiple images sequentially, reusing loaded models."""
        results = []
        for src in sources:
            try:
                results.append(self.run(src))
            except Exception as e:
                log.error("Failed to process %s: %s", src, e)
        return results


# ══════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════
def build_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Advanced Image-to-Text Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("sources", nargs="+", help="Image file(s), URL(s), or base64 string(s)")

    # Preprocessing
    p.add_argument("--no-denoise",    action="store_true")
    p.add_argument("--no-sharpen",    action="store_true")
    p.add_argument("--no-binarise",   action="store_true")
    p.add_argument("--no-deskew",     action="store_true")
    p.add_argument("--binarise-method", default="otsu",
                   choices=["otsu", "adaptive", "sauvola"])
    p.add_argument("--contrast",      type=float, default=1.5)

    # OCR
    p.add_argument("--tesseract-lang",  default="eng")
    p.add_argument("--tesseract-psm",   type=int, default=6)
    p.add_argument("--no-easyocr",      action="store_true")
    p.add_argument("--use-trocr",       action="store_true")
    p.add_argument("--ensemble",        default="confidence",
                   choices=["confidence", "vote", "tess_only"])

    # Post-processing
    p.add_argument("--no-spellcheck",   action="store_true")
    p.add_argument("--detect-entities", action="store_true")

    # Output
    p.add_argument("--formats", nargs="+", default=["txt", "json"],
                   choices=["txt", "json", "pdf"])
    p.add_argument("--output-dir",      default="./output")
    p.add_argument("--workers",         type=int, default=4)

    return p


def main():
    parser = build_cli()
    args   = parser.parse_args()

    config = PipelineConfig(
        output_formats=args.formats,
        output_dir=args.output_dir,
        denoise_strength=0 if args.no_denoise else 10,
        sharpen=not args.no_sharpen,
        binarise=not args.no_binarise,
        binarise_method=args.binarise_method,
        deskew=not args.no_deskew,
        contrast_factor=args.contrast,
        tesseract_lang=args.tesseract_lang,
        tesseract_psm=args.tesseract_psm,
        use_easyocr=not args.no_easyocr,
        use_trocr=args.use_trocr,
        ensemble_strategy=args.ensemble,
        spell_check=not args.no_spellcheck,
        extract_entities=args.detect_entities,
        max_workers=args.workers,
    )

    pipeline = ImageToTextPipeline(config)

    for src in args.sources:
        print(f"\nProcessing: {src}")
        try:
            result = pipeline.run(src)
            print(f"  → {result.total_words if hasattr(result,'total_words') else len(result.full_text.split())} words extracted")
        except Exception as e:
            log.error("Error: %s", e)


if __name__ == "__main__":
    main()