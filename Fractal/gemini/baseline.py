#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gemini Pareidolia Runner (FD12/FD14/FD16) — logging + resume

Place this file at: fractal/gemini/baseline.py

Reads PNGs from ../FD12, ../FD14, ../FD16
Saves:
  - out/gemini_all.jsonl  (streaming, resumable)
  - out/run_gemini.log    (logs)
Optional comparison if ../metadata.csv exists.
"""

import os
import io
import re
import json
import time
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
from PIL import Image

# ---------------- Paths (relative to this file) ----------------
ROOT = Path(__file__).resolve().parent              # fractal/gemini
DATA_ROOT = ROOT.parent                             # fractal/

IMAGE_DIRS = [DATA_ROOT / "FD12", DATA_ROOT / "FD14", DATA_ROOT / "FD16"]
METADATA_CSV = DATA_ROOT / "metadata.csv"          # optional
KEY_PATH = ROOT / "key.txt"                        # fallback if env not set

OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSONL = OUT_DIR / "gemini_all.jsonl"
COMPARE_OUT = ROOT / "compare_out"
LOG_FILE = OUT_DIR / "run_gemini.log"

# ---------------- Model Config ----------------
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
TEMPERATURE = float(os.environ.get("TEMP", "0.0"))
TIMEOUT_SEC = int(os.environ.get("TIMEOUT", "60"))
SLEEP_BETWEEN = float(os.environ.get("SLEEP", "0.10"))
MAX_RETRIES = int(os.environ.get("RETRIES", "3"))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "0.6"))

PROMPT = """You are given an image. Look at the image and determine whether there is a face visible.

If no face is visible, respond with: no face

If a face is visible, respond with the following strict JSON format (and nothing else):

{
  "Hard to spot?": "<Easy|Medium|Hard>",
  "Accident or design?": "<Accident|Design>",
  "Emotion?": "<Happy|Neutral|Disgusted|Angry|Surprised|Scared|Sad|Other>",
  "Person or creature?": "<Human-Adult|Human-Old|Human-Young|Cartoon|Animal|Robot|Alien|Other>",
  "Gender?": "<Male|Female|Neutral>",
  "Amusing?": "<Yes|Somewhat|No>"
}

Do not explain your answer. Respond with either no face or the JSON only.
"""

EXPECTED_ATTRS = [
    "Hard to spot?",
    "Accident or design?",
    "Emotion?",
    "Person or creature?",
    "Gender?",
    "Amusing?",
]

# ---------------- Logging ----------------
def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
                  logging.StreamHandler()],
    )
    logging.info(f"Starting run | MODEL_NAME={MODEL_NAME}")

# ---------------- Helpers ----------------
def read_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if KEY_PATH.exists():
        return KEY_PATH.read_text(encoding="utf-8").strip()
    raise SystemExit("Gemini API key not found. Set GEMINI_API_KEY or provide gemini/key.txt.")

def list_png_images(dirs: List[Path]) -> List[Path]:
    imgs: List[Path] = []
    for d in dirs:
        if d.exists():
            imgs.extend(sorted([p for p in d.glob("*.png") if p.is_file()]))
    return imgs

def reencode_to_jpeg_b64(path: Path) -> Tuple[str, str]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

def safe_json(text: str) -> Optional[dict]:
    t = (text or "").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None

def base_filename(path_str: str) -> str:
    try:
        return Path(path_str).name
    except Exception:
        return str(path_str)

def detect_image_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["image", "file", "filename", "img", "path"])]
    if candidates:
        candidates.sort(key=lambda x: (len(x), x))
        return candidates[0]
    return ""

def normalize_value(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ").replace("/", " ").replace("\\", " ")
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s*-\s*", "-", s)
    synonyms = {
        "yes": "yes", "no": "no", "somewhat": "somewhat",
        "neutral": "neutral", "happy": "happy", "sad": "sad", "other": "other",
        "human-adult": "human-adult", "human adult": "human-adult", "human": "human",
        "cartoon": "cartoon", "male": "male", "female": "female",
        "easy": "easy", "medium": "medium", "hard": "hard",
        "accident": "accident", "design": "design",
    }
    return synonyms.get(s, s)

def load_done_set(jsonl_path: Path) -> set:
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            img = obj.get("image_path") or obj.get("image") or obj.get("path")
            if img:
                done.add(base_filename(img))
    return done

def append_jsonl(jsonl_path: Path, row: dict) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---------------- Gemini call ----------------
def gemini_generate(model, prompt: str, b64: str, mime: str, timeout: int) -> Tuple[str, int]:
    t0 = time.time()
    resp = model.generate_content(
        [
            {"text": prompt},
            {"inline_data": {"mime_type": mime, "data": b64}},
        ],
        generation_config={
            "temperature": TEMPERATURE,
            "response_mime_type": "application/json",
        },
        request_options={"timeout": timeout},
    )
    latency_ms = int((time.time() - t0) * 1000)
    return resp.text, latency_ms

# ---------------- Main ----------------
def main():
    setup_logger()

    # Configure Gemini client
    api_key = read_key()
    import google.generativeai as genai
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel(MODEL_NAME)

    # Discover images & resume
    images = list_png_images(IMAGE_DIRS)
    if not images:
        logging.error(f"No PNG images found under: {', '.join(str(d) for d in IMAGE_DIRS)}")
        return
    done = load_done_set(RESULTS_JSONL)
    todo = [p for p in images if p.name not in done]
    logging.info(f"Discovered {len(images)} PNG | Done={len(done)} | Pending={len(todo)}")

    processed = 0
    for idx, p in enumerate(todo, 1):
        logging.info(f"[{idx}/{len(todo)}] {p.name} — encode")
        b64, mime = reencode_to_jpeg_b64(p)

        attempt = 0
        row = None
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                logging.info(f"[{idx}/{len(todo)}] {p.name} — call attempt {attempt}")
                text, latency_ms = gemini_generate(model, PROMPT, b64, mime, TIMEOUT_SEC)

                if text.strip().lower() == "no face":
                    row = {
                        "image_path": str(p),
                        "model": MODEL_NAME,
                        "latency_ms": latency_ms,
                        "error": "no_face",
                        "_raw": text,
                    }
                    logging.info(f"[{idx}/{len(todo)}] {p.name} — OK no_face | {latency_ms}ms")
                else:
                    data = safe_json(text)
                    if data is None:
                        row = {
                            "image_path": str(p),
                            "model": MODEL_NAME,
                            "latency_ms": latency_ms,
                            "error": "non_json_response",
                            "_raw": text,
                        }
                        logging.warning(f"[{idx}/{len(todo)}] {p.name} — non_json_response | {latency_ms}ms")
                    else:
                        for attr in EXPECTED_ATTRS:
                            data.setdefault(attr, None)
                        row = {"image_path": str(p), "model": MODEL_NAME, "latency_ms": latency_ms, **data}
                        logging.info(f"[{idx}/{len(todo)}] {p.name} — OK | {latency_ms}ms")
                break

            except Exception as e:
                logging.warning(f"[{idx}/{len(todo)}] {p.name} — attempt {attempt} failed: {e}")
                if attempt >= MAX_RETRIES:
                    row = {"image_path": str(p), "model": MODEL_NAME, "error": f"exception:{e}"}
                    break
                sleep_s = min(8.0, BACKOFF_BASE * (2 ** (attempt - 1)))
                time.sleep(sleep_s)

        append_jsonl(RESULTS_JSONL, row)
        processed += 1
        time.sleep(SLEEP_BETWEEN)

    logging.info(f"Completed {processed} images. Results: {RESULTS_JSONL}")

    # Optional comparison
    if not METADATA_CSV.exists():
        logging.info(f"{METADATA_CSV} not found. Skipping comparison.")
        return

    rows: List[dict] = []
    with open(RESULTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        logging.warning("No rows parsed from results JSONL. Exiting.")
        return

    df = pd.DataFrame(rows)
    if "error" in df.columns:
        mask_err = df["error"].astype(str).str.len().gt(0) & df["error"].notna()
    else:
        mask_err = pd.Series(False, index=df.index)

    errs = df.loc[mask_err].copy()
    df_good = df.loc[~mask_err].copy()
    if "image_path" not in df_good.columns or df_good.empty:
        logging.warning("No valid predictions to score.")
        return
    df_good["_image_file"] = df_good["image_path"].map(base_filename)

    meta = pd.read_csv(METADATA_CSV)
    img_col = detect_image_column(meta)
    if not img_col:
        for c in ["image", "filename", "file", "img", "path", "Image", "Filename"]:
            if c in meta.columns:
                img_col = c
                break
    if not img_col:
        logging.warning("Could not detect the image filename column in metadata.csv. Skipping comparison.")
        return
    meta["_image_file"] = meta[img_col].apply(base_filename)

    def map_cols(dfcols):
        mapping = {}
        norm = {re.sub(r"[^\w]+", "", c.lower()): c for c in dfcols}
        for exp in EXPECTED_ATTRS:
            key = re.sub(r"[^\w]+", "", exp.lower())
            mapping[exp] = norm.get(key)
        return mapping

    colmap = map_cols(meta.columns)
    keep_cols = ["_image_file"] + [c for c in colmap.values() if c is not None]
    meta_small = meta[keep_cols].copy()
    rename_map = {v: k for k, v in colmap.items() if v is not None}
    meta_small = meta_small.rename(columns=rename_map)
    for attr in EXPECTED_ATTRS:
        if attr not in meta_small.columns:
            meta_small[attr] = None

    merged = pd.merge(df_good, meta_small, on="_image_file", how="inner", suffixes=("_pred", "_gt"))

    summary_rows = []
    mismatches = []
    for attr in EXPECTED_ATTRS:
        pred_col = f"{attr}_pred"
        gt_col = f"{attr}_gt"
        if pred_col not in merged.columns or gt_col not in merged.columns:
            continue

        sub = merged[["_image_file", pred_col, gt_col]].copy()
        sub["pred_norm"] = sub[pred_col].map(normalize_value)
        sub["gt_norm"] = sub[gt_col].map(normalize_value)
        sub["match"] = (sub["pred_norm"] == sub["gt_norm"]) & sub["gt_norm"].ne("")

        total = int(sub["gt_norm"].ne("").sum())
        correct = int(sub["match"].sum())
        acc = (correct / total) if total else 0.0

        summary_rows.append(
            {"attribute": attr, "n_scored": total, "n_correct": correct, "accuracy": round(acc, 4)}
        )

        mm = sub[~sub["match"]].copy()
        mm = mm.rename(columns={pred_col: "pred_raw", gt_col: "gt_raw"})
        mm = mm[["_image_file", "pred_raw", "gt_raw", "pred_norm", "gt_norm"]]
        mm.insert(0, "attribute", attr)
        mismatches.append(mm)

    summary = pd.DataFrame(summary_rows).sort_values("attribute")
    mismatch_df = pd.concat(mismatches, ignore_index=True) if mismatches else pd.DataFrame()

    COMPARE_OUT.mkdir(parents=True, exist_ok=True)
    summary_path = COMPARE_OUT / "summary.csv"
    mismatches_path = COMPARE_OUT / "mismatches_head.csv"
    merged_sample_path = COMPARE_OUT / "merged_sample.csv"
    errors_path = COMPARE_OUT / "skipped_errors.jsonl"

    summary.to_csv(summary_path, index=False)
    mismatch_df.head(200).to_csv(mismatches_path, index=False)
    merged.head(100).to_csv(merged_sample_path, index=False)

    if not errs.empty:
        with open(errors_path, "w", encoding="utf-8") as f:
            for _, r in errs.iterrows():
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    logging.info("=== Comparison summary (ALL images) ===")
    if summary.empty:
        logging.info("No comparable attributes found. Ensure prompt keys match metadata column names exactly.")
    else:
        logging.info("\n" + summary.to_string(index=False))

    if not mismatch_df.empty:
        logging.info("Saved mismatch head to: %s", mismatches_path)
    logging.info("Saved reports:\n - %s\n - %s\n - %s", summary_path, mismatches_path, merged_sample_path)
    logging.info("Done.")

if __name__ == "__main__":
    main()
