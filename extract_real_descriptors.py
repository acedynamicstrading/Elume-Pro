"""
ÉLUME Pro — Real Descriptor Extractor
Scans every image in recommend/, extracts 6 geometric face descriptors
using MediaPipe FaceLandmarker, and maps each to its tryon/ counterpart.

OUTPUT: style_db_descriptors.json  (drop this next to index.html)

DESCRIPTOR DEFINITIONS (all landmarks from MediaPipe 468-point FaceMesh):
  lm[10]  = forehead crown
  lm[152] = chin bottom
  lm[33]  = right eye outer corner
  lm[263] = left  eye outer corner
  lm[234] = right temple / ear-connection
  lm[454] = left  temple / ear-connection
  lm[172] = right lower jaw corner
  lm[397] = left  lower jaw corner
  lm[93]  = right upper-jaw / cheekbone
  lm[323] = left  upper-jaw / cheekbone

COMPUTED:
  face_height  = |lm[152].y − lm[10].y|
  eye_span     = |lm[263].x − lm[33].x|
  temple_span  = |lm[454].x − lm[234].x|
  jaw_span     = |lm[397].x − lm[172].x|   (lower jaw)
  upper_jaw    = |lm[323].x − lm[93].x|    (upper jaw / cheekbone)

SIX DESCRIPTORS (same formula used in scanner.html):
  aspect       = face_height / eye_span         (elongation)
  jaw_ratio    = jaw_span    / temple_span       (jaw vs temple)
  temple_ratio = temple_span / eye_span          (temple vs eye)
  chin_taper   = jaw_span    / upper_jaw         (jaw narrowing)
  rx           = eye_span    / 2                 (norm half-width)
  ry           = face_height / 2                 (norm half-height)

USAGE:
  python extract_real_descriptors.py
  python extract_real_descriptors.py --recommend ./recommend --tryon ./tryon
  python extract_real_descriptors.py --out custom_output.json --workers 4
"""

import json
import os
import re
import sys
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"\n❌  Missing dependency: {e}")
    print("   Run:  pip install mediapipe numpy tqdm --break-system-packages\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/latest/face_landmarker.task")
MODEL_PATH = "face_landmarker.task"

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Landmark indices (MediaPipe 468-point canonical face model)
IDX = dict(
    crown        = 10,
    chin         = 152,
    eye_r_outer  = 33,    # right eye outer corner
    eye_l_outer  = 263,   # left  eye outer corner
    temple_r     = 234,   # right ear / temple
    temple_l     = 454,   # left  ear / temple
    jaw_r        = 172,   # lower jaw right
    jaw_l        = 397,   # lower jaw left
    upper_jaw_r  = 93,    # upper jaw / cheekbone right
    upper_jaw_l  = 323,   # upper jaw / cheekbone left
)

# Weighted matching used in index.html (kept here for reference)
WEIGHTS = dict(aspect=2.0, jaw_ratio=1.8, temple_ratio=1.2,
               chin_taper=1.5, rx=0.5, ry=0.5)


# ─────────────────────────────────────────────────────────────
#  MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────
def ensure_model(path: str = MODEL_PATH) -> str:
    if not os.path.exists(path):
        print(f"📥  Downloading FaceLandmarker model → {path}")
        try:
            urllib.request.urlretrieve(MODEL_URL, path)
            print("    ✓ Download complete")
        except Exception as e:
            print(f"❌  Download failed: {e}")
            print(f"   Manually download:\n   {MODEL_URL}")
            sys.exit(1)
    return path


# ─────────────────────────────────────────────────────────────
#  DETECTOR FACTORY  (one per thread)
# ─────────────────────────────────────────────────────────────
def make_detector(model_path: str):
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        num_faces=1,
        min_face_detection_confidence=0.05,
        min_face_presence_confidence=0.05,
        min_tracking_confidence=0.05,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


# ─────────────────────────────────────────────────────────────
#  DESCRIPTOR COMPUTATION  — must stay byte-for-byte identical
#  to the JavaScript version in scanner.html / head-scanner.html
# ─────────────────────────────────────────────────────────────
def compute_descriptors(landmarks) -> dict | None:
    """
    landmarks: list of NormalizedLandmark (x, y, z all in 0-1 range)
    Returns dict with 6 descriptors or None if geometry is invalid.
    """
    if len(landmarks) < 468:
        return None

    def x(i): return landmarks[i].x
    def y(i): return landmarks[i].y

    face_height   = abs(y(IDX['chin'])        - y(IDX['crown']))
    eye_span      = abs(x(IDX['eye_l_outer']) - x(IDX['eye_r_outer']))
    temple_span   = abs(x(IDX['temple_l'])    - x(IDX['temple_r']))
    jaw_span      = abs(x(IDX['jaw_l'])       - x(IDX['jaw_r']))
    upper_jaw_span= abs(x(IDX['upper_jaw_l']) - x(IDX['upper_jaw_r']))

    # Guard against degenerate detections
    if eye_span < 0.05 or face_height < 0.05 or temple_span < 0.05:
        return None
    if upper_jaw_span < 0.02:
        upper_jaw_span = jaw_span * 0.95   # fallback

    aspect       = round(face_height   / eye_span,       4)
    jaw_ratio    = round(jaw_span      / temple_span,     4)
    temple_ratio = round(temple_span   / eye_span,        4)
    chin_taper   = round(jaw_span      / upper_jaw_span,  4)
    rx           = round(eye_span      / 2,               4)
    ry           = round(face_height   / 2,               4)

    # Sanity ranges — widened for hairstyle images (wider temple ratios than portrait shots)
    if not (0.8  < aspect       < 3.0):  return None
    if not (0.3  < jaw_ratio    < 1.8):  return None
    if not (0.5  < temple_ratio < 2.2):  return None
    if not (0.2  < chin_taper   < 2.0):  return None
    if not (0.03 < rx           < 0.6):  return None
    if not (0.05 < ry           < 0.7):  return None

    return dict(aspect=aspect, jaw_ratio=jaw_ratio, temple_ratio=temple_ratio,
                chin_taper=chin_taper, rx=rx, ry=ry)


# ─────────────────────────────────────────────────────────────
#  TRYON FILE MAPPING
# ─────────────────────────────────────────────────────────────
def build_tryon_map(tryon_dir: Path) -> dict[str, str]:
    """
    Maps recommend-image basenames → matching tryon PNG filenames.
    Convention:  <recommend_base>-<category>.png
    E.g.  alicia-silverstone-long-wavy-hairstyle.jpg.jpg
       → alicia-silverstone-long-wavy-hairstyle.jpg-long-wavy.png
    The base = recommend filename stripped of final extension.
    """
    tryon_map: dict[str, str] = {}
    if not tryon_dir.is_dir():
        return tryon_map

    tryon_files = [f.name for f in tryon_dir.iterdir() if f.suffix.lower() == ".png"]

    for tfile in tryon_files:
        # Strip the category suffix to get the image base
        # e.g.  foo.jpg-long-wavy.png  →  base = "foo.jpg"
        #        bar-short-wavy-male.png → base = "bar"
        base = re.sub(r'-(?:short|medium|long)-.*$', '', tfile.replace('.png', ''))
        if base not in tryon_map:
            tryon_map[base] = tfile

    return tryon_map


def find_tryon(img_name: str, tryon_map: dict[str, str]) -> str | None:
    """
    Tries several strategies to find the tryon file for an image.
    img_name is the recommend/ filename (may have double extension like .jpg.jpg)
    """
    # Strategy 1: strip final extension → lookup
    stem = img_name.rsplit(".", 1)[0] if "." in img_name else img_name
    if stem in tryon_map:
        return tryon_map[stem]

    # Strategy 2: full name without .jpg or .png
    for key in tryon_map:
        if img_name.startswith(key):
            return tryon_map[key]

    # Strategy 3: fuzzy — key is a prefix of stem
    for key in tryon_map:
        if stem.startswith(key) or key.startswith(stem[:20]):
            return tryon_map[key]

    return None


# ─────────────────────────────────────────────────────────────
#  GENDER DETECTION FROM FILENAME
# ─────────────────────────────────────────────────────────────
MALE_TOKENS = {"male", "man", "men", "boy", "him", "his", "mr",
               "handsome", "bearded", "businessman", "dude"}

def is_male(filename: str) -> bool:
    parts = set(re.split(r'[-_.\s]+', filename.lower()))
    return bool(parts & MALE_TOKENS)


# ─────────────────────────────────────────────────────────────
#  PROCESS ONE IMAGE
# ─────────────────────────────────────────────────────────────
def load_as_mp_image(img_path: Path):
    """Load image as MediaPipe Image, with PIL fallback and upscaling."""
    from PIL import Image as PILImage
    try:
        pil_img = PILImage.open(img_path).convert("RGB")
    except Exception:
        return None
    # Upscale if face is likely too small (image shorter than 480px)
    w, h = pil_img.size
    min_dim = min(w, h)
    if min_dim < 480:
        scale = 480 / min_dim
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    arr = np.array(pil_img, dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)


def process_image(img_path: Path, detector, tryon_map: dict) -> tuple[str, dict | None]:
    """Returns (filename, descriptor_dict | None)."""
    try:
        mp_image = load_as_mp_image(img_path)
        if mp_image is None:
            return img_path.name, {"_error": "Failed to load image from file"}

        result = detector.detect(mp_image)

        if not result.face_landmarks:
            return img_path.name, None

        landmarks = result.face_landmarks[0]
        desc = compute_descriptors(landmarks)
        if desc is None:
            return img_path.name, None

        desc["male"]      = is_male(img_path.name)
        desc["tryonFile"] = find_tryon(img_path.name, tryon_map)
        return img_path.name, desc

    except Exception as e:
        return img_path.name, {"_error": str(e)}


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ÉLUME descriptor extractor")
    parser.add_argument("--recommend", default="./recommend",
                        help="Path to recommend/ image folder")
    parser.add_argument("--tryon", default="./tryon",
                        help="Path to tryon/ folder")
    parser.add_argument("--out", default="style_db_descriptors.json",
                        help="Output JSON filename")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (keep at 1 if RAM is limited)")
    args = parser.parse_args()

    rec_dir   = Path(args.recommend)
    tryon_dir = Path(args.tryon)
    out_path  = Path(args.out)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   ÉLUME Pro — Real Descriptor Extractor                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    if not rec_dir.is_dir():
        print(f"❌  recommend/ dir not found: {rec_dir}")
        sys.exit(1)

    # Collect images
    images = sorted([p for p in rec_dir.iterdir()
                     if p.suffix.lower() in IMG_EXTS])
    print(f"📁  Found {len(images)} images in {rec_dir}")

    # Build tryon map
    tryon_map = build_tryon_map(tryon_dir)
    print(f"🎯  Tryon map: {len(tryon_map)} entries found in {tryon_dir}")

    # Ensure model
    model_path = ensure_model()
    print()

    results: dict[str, dict] = {}
    errors:  list[str] = []
    no_face: list[str] = []

    # ── single-threaded path (most reliable with MediaPipe) ──
    detector = make_detector(model_path)

    for img_path in tqdm(images, desc="Extracting", unit="img"):
        name, desc = process_image(img_path, detector, tryon_map)
        if desc is None:
            no_face.append(name)
        elif "_error" in desc:
            errors.append(f"{name}: {desc['_error']}")
        else:
            results[name] = desc

    # ── write output ──
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ── summary ──
    print()
    print("═" * 64)
    print(f"✅  Extracted:   {len(results):>4} descriptors")
    print(f"⚠️   No face:     {len(no_face):>4} images  (no landmarks detected)")
    print(f"❌  Errors:      {len(errors):>4} images")
    print(f"📄  Output:      {out_path}")
    print("═" * 64)

    if no_face:
        print(f"\nImages where no face was found ({len(no_face)}):")
        for n in no_face[:20]:
            print(f"  • {n}")
        if len(no_face) > 20:
            print(f"  … and {len(no_face)-20} more")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  ⚠  {e}")

    print()
    print("Next steps:")
    print("  1. Copy style_db_descriptors.json next to index.html")
    print("  2. Validate:  python validate_descriptors.py style_db_descriptors.json")
    print("  3. Deploy to GitHub Pages")
    print()


if __name__ == "__main__":
    main()
