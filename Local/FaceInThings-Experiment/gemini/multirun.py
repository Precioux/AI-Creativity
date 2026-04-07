#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_and_compare.py

End-to-end script to process ALL images in FacesInThings/images with Gemini,
save streaming JSONL outputs (with resume), and compare against metadata.csv.

REPETITION MODE:
  Each image is queried N_REPS times independently (sampling the model stochastically
  or verifying stability at TEMPERATURE=0). Per-attribute accuracy is computed per rep,
  then mean ± std is reported across all reps. A majority-vote prediction per image is
  also computed and scored as the "ensemble" accuracy.

What it does:
1) Reads Gemini API key from key.txt (PATHS section below).
2) Iterates over ALL images × ALL rep indices (with resume).
3) For each (image, rep), requests EXACT six attributes as JSON (dataset-aligned).
4) Appends one line per (image, rep) to out/gemini_all.jsonl as soon as it's ready.
5) After completion, loads metadata.csv and computes:
     - per-rep accuracy table
     - mean ± std across reps per attribute
     - majority-vote (ensemble) accuracy per attribute
6) Writes comparison reports to compare_out/ and prints a summary.

You can safely stop & rerun; it will skip already-processed (image, rep) pairs.
"""

import os
import io
import re
import json
import time
import base64
import statistics
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import Counter

import pandas as pd
from PIL import Image

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),              # terminal
        logging.FileHandler("multirun.log", mode="a"),  # persistent log file
    ],
)
log = logging.getLogger(__name__)

# ====================== PATHS — EDIT IF NEEDED ======================
GEMINI_FOLDER   = Path("/Users/precioux/Desktop/projects/Creativity/code/gemini")
KEY_PATH        = GEMINI_FOLDER / "key.txt"

IMAGES_DIR      = Path("/Users/precioux/Desktop/projects/Creativity/FacesInThings/images")
METADATA_CSV    = Path("/Users/precioux/Desktop/projects/Creativity/FacesInThings/metadata.csv")

OUT_BASE        = GEMINI_FOLDER
RESULTS_JSONL   = OUT_BASE / "multirun-out/gemini_all.jsonl"
COMPARE_OUT     = OUT_BASE / "multirun-compare-out"

# ====================== MODEL CONFIG ======================
MODEL_NAME    = "gemini-2.5-flash-lite"   # swap to "gemini-1.5-flash" if needed
TEMPERATURE   = 1.0    # Set > 0 for meaningful repetition variance; 0.0 = stable but redundant
N_REPS        = 50     # Number of independent repetitions per image (50–100 recommended)
TIMEOUT_SEC   = 60
SLEEP_BETWEEN = 0.10   # polite spacing between API calls (seconds)
MAX_RETRIES   = 3

# Ask for the six Faces-in-Things attributes ONLY (aligned to dataset categories)
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


# ====================== HELPERS ======================
def ensure_dirs():
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    COMPARE_OUT.mkdir(parents=True, exist_ok=True)


def read_key(key_path: Path) -> str:
    return key_path.read_text(encoding="utf-8").strip()


def reencode_to_jpeg_b64(path: Path) -> Tuple[str, str]:
    """Open an image and re-encode as JPEG to standardize input to the VLM."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def list_all_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])


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
    s = s.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
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
    """
    Return set of (basename, rep_index) tuples already processed in JSONL.
    Backwards-compatible: rows without a 'rep' field are treated as rep=0.
    """
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
            rep = obj.get("rep", 0)
            if img:
                done.add((base_filename(img), int(rep)))
    return done


def append_jsonl(jsonl_path: Path, row: dict):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def majority_vote(values: List[str]) -> str:
    """Return the most common non-empty normalized value in a list."""
    cleaned = [v for v in values if v and v.strip()]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


# ====================== MAIN ======================
def main():
    ensure_dirs()

    log.info("=" * 60)
    log.info("multirun.py starting")
    log.info(f"  Model      : {MODEL_NAME}")
    log.info(f"  N_REPS     : {N_REPS}")
    log.info(f"  TEMPERATURE: {TEMPERATURE}")
    log.info(f"  IMAGES_DIR : {IMAGES_DIR}")
    log.info(f"  RESULTS    : {RESULTS_JSONL}")
    log.info("=" * 60)

    # 1) Configure Gemini
    log.info("Reading API key...")
    api_key = read_key(KEY_PATH)
    log.info(f"API key loaded (length={len(api_key)})")

    import google.generativeai as genai
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel(MODEL_NAME)
    log.info(f"Gemini model initialised: {MODEL_NAME}")

    # 2) Gather all images and resume state
    log.info(f"Scanning images in: {IMAGES_DIR}")
    images = list_all_images(IMAGES_DIR)
    if not images:
        log.error(f"No images found in {IMAGES_DIR}")
        raise SystemExit(f"No images found in {IMAGES_DIR}")
    log.info(f"Found {len(images)} images")

    log.info(f"Loading resume state from: {RESULTS_JSONL}")
    done = load_done_set(RESULTS_JSONL)
    log.info(f"Already completed: {len(done)} (image, rep) pairs")

    todo = [
        (p, rep)
        for p in images
        for rep in range(N_REPS)
        if (base_filename(str(p)), rep) not in done
    ]

    total_work = len(images) * N_REPS
    log.info(f"Total work units : {total_work}  ({len(images)} images × {N_REPS} reps)")
    log.info(f"Remaining to do  : {len(todo)}")

    if not todo:
        log.info("Nothing to do — all (image, rep) pairs already processed.")
        log.info("Skipping to comparison step...")

    # 3) Process all pending (image, rep) pairs
    processed  = 0
    n_success  = 0
    n_noface   = 0
    n_parse_err = 0
    n_api_err  = 0

    for idx, (p, rep) in enumerate(todo, 1):
        log.debug(f"[{idx}/{len(todo)}] Encoding image: {p.name}  rep={rep}")
        b64, mime = reencode_to_jpeg_b64(p)
        log.debug(f"  → JPEG encoded, mime={mime}")

        attempt = 0
        row     = None

        while attempt < MAX_RETRIES:
            log.debug(f"  → Sending to Gemini (attempt {attempt + 1}/{MAX_RETRIES})...")
            try:
                t0 = time.time()
                resp = model.generate_content(
                    [
                        {"text": PROMPT},
                        {"inline_data": {"mime_type": mime, "data": b64}},
                    ],
                    generation_config={
                        "temperature": TEMPERATURE,
                        "response_mime_type": "application/json",
                    },
                    request_options={"timeout": TIMEOUT_SEC},
                )
                latency_ms = int((time.time() - t0) * 1000)
                raw_text   = resp.text.strip()

                log.debug(f"  ← Response received in {latency_ms} ms")
                log.debug(f"  ← Raw response: {raw_text[:300]}")   # cap at 300 chars

                # ---- Detect "no face" plain-text response ----
                if raw_text.lower().startswith("no face"):
                    log.info(f"  ✗ NO FACE detected | {p.name} | rep={rep} | {latency_ms} ms")
                    row = {
                        "image_path": str(p),
                        "rep": rep,
                        "model": MODEL_NAME,
                        "latency_ms": latency_ms,
                        "no_face": True,
                        **{attr: None for attr in EXPECTED_ATTRS},
                    }
                    n_noface += 1
                    break

                data = safe_json(raw_text)

                if data is None:
                    log.warning(f"  ⚠ JSON parse FAILED | {p.name} | rep={rep}")
                    log.warning(f"    Raw text was: {raw_text[:200]}")
                    row = {
                        "image_path": str(p),
                        "rep": rep,
                        "model": MODEL_NAME,
                        "latency_ms": latency_ms,
                        "error": "non_json_response",
                        "_raw": raw_text,
                    }
                    n_parse_err += 1
                else:
                    for attr in EXPECTED_ATTRS:
                        data.setdefault(attr, None)

                    # Log every predicted attribute value
                    attr_summary = " | ".join(
                        f"{k.split('?')[0].strip()}={v}" for k, v in data.items() if k in EXPECTED_ATTRS
                    )
                    log.info(
                        f"  ✓ OK  [{idx:>6}/{len(todo)}] rep={rep:03d} | {latency_ms:>5} ms "
                        f"| {p.name[:40]:<40} | {attr_summary}"
                    )
                    row = {
                        "image_path": str(p),
                        "rep": rep,
                        "model": MODEL_NAME,
                        "latency_ms": latency_ms,
                        **data,
                    }
                    n_success += 1
                break

            except Exception as e:
                attempt += 1
                log.warning(f"  ⚠ API exception (attempt {attempt}): {e}")
                if attempt >= MAX_RETRIES:
                    log.error(f"  ✗ FAILED after {MAX_RETRIES} attempts | {p.name} | rep={rep}")
                    log.error(f"    Exception: {e}")
                    row = {
                        "image_path": str(p),
                        "rep": rep,
                        "model": MODEL_NAME,
                        "error": f"exception:{e}",
                    }
                    n_api_err += 1
                else:
                    backoff = 0.5 * attempt
                    log.debug(f"  → Backing off {backoff:.1f}s before retry...")
                    time.sleep(backoff)

        append_jsonl(RESULTS_JSONL, row)
        processed += 1

        # Progress milestone every 50 units
        if processed % 50 == 0 or processed == len(todo):
            pct = 100 * processed / len(todo) if todo else 100
            log.info(
                f"── PROGRESS {pct:5.1f}% ── {processed}/{len(todo)} done "
                f"| ✓ ok={n_success} ✗ noface={n_noface} "
                f"⚠ parse_err={n_parse_err} ✗ api_err={n_api_err}"
            )

        time.sleep(SLEEP_BETWEEN)

    log.info(f"\nAll processing complete.")
    log.info(f"  Successful predictions : {n_success}")
    log.info(f"  No-face responses      : {n_noface}")
    log.info(f"  JSON parse errors      : {n_parse_err}")
    log.info(f"  API / timeout errors   : {n_api_err}")
    log.info(f"  Results saved to       : {RESULTS_JSONL}")

    # =========================================================
    # 4) COMPARISON — Load ALL results (all reps) into DataFrame
    # =========================================================
    rows = []
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
        log.error("No rows parsed from results JSONL. Exiting.")
        return

    df = pd.DataFrame(rows)
    df["rep"] = df.get("rep", 0).fillna(0).astype(int) if "rep" in df.columns else 0

    # Split errors
    if "error" in df.columns:
        mask_err = df["error"].astype(str).str.len().gt(0) & df["error"].notna()
    else:
        mask_err = pd.Series(False, index=df.index)

    errs    = df.loc[mask_err].copy()
    df_good = df.loc[~mask_err].copy()

    if "image_path" not in df_good.columns or df_good.empty:
        log.error("No valid predictions to score (all lines errored or missing image_path).")
        return

    df_good["_image_file"] = df_good["image_path"].map(base_filename)

    # 5) Load metadata & align columns
    meta    = pd.read_csv(METADATA_CSV)
    img_col = detect_image_column(meta)
    if not img_col:
        for c in ["image", "filename", "file", "img", "path", "Image", "Filename"]:
            if c in meta.columns:
                img_col = c
                break
    if not img_col:
        raise SystemExit("Could not detect the image filename column in metadata.csv.")

    meta["_image_file"] = meta[img_col].apply(base_filename)

    def map_cols(dfcols):
        norm = {re.sub(r"[^\w]+", "", c.lower()): c for c in dfcols}
        return {exp: norm.get(re.sub(r"[^\w]+", "", exp.lower())) for exp in EXPECTED_ATTRS}

    colmap    = map_cols(meta.columns)
    keep_cols = ["_image_file"] + [c for c in colmap.values() if c is not None]
    meta_small = meta[keep_cols].copy()
    meta_small = meta_small.rename(columns={v: k for k, v in colmap.items() if v is not None})
    for attr in EXPECTED_ATTRS:
        if attr not in meta_small.columns:
            meta_small[attr] = None

    # =========================================================
    # 6) Per-rep accuracy  (images that appear in metadata only)
    # =========================================================
    merged = pd.merge(df_good, meta_small, on="_image_file", how="inner", suffixes=("_pred", "_gt"))

    # Normalize all prediction and GT columns once
    for attr in EXPECTED_ATTRS:
        pc = f"{attr}_pred"
        gc = f"{attr}_gt"
        if pc in merged.columns:
            merged[f"{attr}_pred_norm"] = merged[pc].map(normalize_value)
        if gc in merged.columns:
            merged[f"{attr}_gt_norm"] = merged[gc].map(normalize_value)

    # Per-rep accuracy table  → shape: (N_REPS, len(EXPECTED_ATTRS))
    rep_acc_rows = []
    for rep_idx, grp in merged.groupby("rep"):
        row = {"rep": rep_idx}
        for attr in EXPECTED_ATTRS:
            pn = f"{attr}_pred_norm"
            gn = f"{attr}_gt_norm"
            if pn not in grp.columns or gn not in grp.columns:
                row[attr] = float("nan")
                continue
            sub   = grp[[pn, gn]].copy()
            valid = sub[gn].ne("")
            if valid.sum() == 0:
                row[attr] = float("nan")
            else:
                row[attr] = float((sub.loc[valid, pn] == sub.loc[valid, gn]).sum()) / valid.sum()
        rep_acc_rows.append(row)

    rep_acc_df = pd.DataFrame(rep_acc_rows).sort_values("rep").reset_index(drop=True)

    # =========================================================
    # 7) Aggregate: mean ± std across reps
    # =========================================================
    summary_rows = []
    for attr in EXPECTED_ATTRS:
        if attr not in rep_acc_df.columns:
            continue
        vals  = rep_acc_df[attr].dropna().tolist()
        n_reps_done = len(vals)
        mean_ = statistics.mean(vals) if vals else float("nan")
        std_  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append({
            "attribute"   : attr,
            "n_reps"      : n_reps_done,
            "mean_acc"    : round(mean_, 4),
            "std_acc"     : round(std_,  4),
            "min_acc"     : round(min(vals), 4) if vals else float("nan"),
            "max_acc"     : round(max(vals), 4) if vals else float("nan"),
        })

    summary_mean = pd.DataFrame(summary_rows).sort_values("attribute")

    # =========================================================
    # 8) Majority-vote (ensemble) accuracy
    # =========================================================
    mv_rows = []
    for attr in EXPECTED_ATTRS:
        pn = f"{attr}_pred_norm"
        gn = f"{attr}_gt_norm"
        if pn not in merged.columns or gn not in merged.columns:
            continue
        # aggregate per image: majority vote across reps
        agg = (
            merged[["_image_file", pn, gn]]
            .groupby("_image_file")
            .agg(
                mv_pred = (pn, majority_vote),
                gt      = (gn, "first"),      # GT is constant across reps
            )
            .reset_index()
        )
        valid    = agg["gt"].ne("")
        total    = int(valid.sum())
        correct  = int((agg.loc[valid, "mv_pred"] == agg.loc[valid, "gt"]).sum())
        mv_acc   = correct / total if total else float("nan")
        mv_rows.append({
            "attribute"     : attr,
            "n_images"      : total,
            "mv_correct"    : correct,
            "mv_accuracy"   : round(mv_acc, 4),
        })

    mv_summary = pd.DataFrame(mv_rows).sort_values("attribute")

    # =========================================================
    # 9) Save reports
    # =========================================================
    COMPARE_OUT.mkdir(parents=True, exist_ok=True)

    rep_acc_path      = COMPARE_OUT / "per_rep_accuracy.csv"
    mean_summary_path = COMPARE_OUT / "mean_std_summary.csv"
    mv_summary_path   = COMPARE_OUT / "majority_vote_summary.csv"
    merged_path       = COMPARE_OUT / "merged_sample.csv"
    errors_path       = COMPARE_OUT / "skipped_errors.jsonl"

    rep_acc_df.to_csv(rep_acc_path,     index=False)
    summary_mean.to_csv(mean_summary_path, index=False)
    mv_summary.to_csv(mv_summary_path,  index=False)
    merged.head(200).to_csv(merged_path, index=False)

    if not errs.empty:
        with open(errors_path, "w", encoding="utf-8") as f:
            for _, r in errs.iterrows():
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # =========================================================
    # 10) Log concise summary
    # =========================================================
    log.info(f"\n=== Per-rep accuracy (head, {min(5, len(rep_acc_df))} of {len(rep_acc_df)} reps) ===")
    log.info("\n" + rep_acc_df.head(5).to_string(index=False))

    log.info(f"\n=== Mean ± Std accuracy across {N_REPS} reps ===")
    if summary_mean.empty:
        log.warning("No comparable attributes found.")
    else:
        log.info("\n" + summary_mean.to_string(index=False))

    log.info("\n=== Majority-vote (ensemble) accuracy ===")
    if mv_summary.empty:
        log.warning("No results.")
    else:
        log.info("\n" + mv_summary.to_string(index=False))

    if not errs.empty:
        log.warning(f"{len(errs)} result lines with errors → {errors_path}")

    log.info(f"\nSaved reports:")
    log.info(f"  Per-rep accuracies : {rep_acc_path}")
    log.info(f"  Mean ± Std summary : {mean_summary_path}")
    log.info(f"  Majority-vote      : {mv_summary_path}")
    log.info(f"  Merged sample      : {merged_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()