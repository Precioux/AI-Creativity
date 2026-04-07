#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_and_compare_qwen.py

End-to-end script to process ALL images in FacesInThings/images with Qwen (via Ollama),
save JSONL outputs (with resume), and compare against metadata.csv.
Also prints each image's output to stdout.

You can safely stop & rerun; it will skip already-processed images.
"""

import os
import io
import re
import json
import time
import base64
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import requests
import pandas as pd
from PIL import Image

# ====================== PATHS — EDIT IF NEEDED ======================
BASE_FOLDER   = Path("/mnt/mahdipou/test/code/qwen")  # keep your code folder
IMAGES_DIR    = Path("/mnt/mahdipou/test/FacesInThings/images")
METADATA_CSV  = Path("/mnt/mahdipou/test/FacesInThings/metadata.csv")

OUT_BASE      = BASE_FOLDER
RESULTS_JSONL = OUT_BASE / "out/qwen_all.jsonl"
COMPARE_OUT   = OUT_BASE / "compare_out"

# ====================== OLLAMA / MODEL CONFIG ======================
OLLAMA_BASE   = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")

# IMPORTANT: matches your installed tag (shown by `ollama list`)
MODEL_TAG     = os.environ.get("QWEN_TAG") or "qwen2.5vl"  # <-- changed (no dash)

TEMPERATURE   = 0.0
TIMEOUT_SEC   = 120
SLEEP_BETWEEN = 0.10
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
    (OUT_BASE / "out").mkdir(parents=True, exist_ok=True)
    COMPARE_OUT.mkdir(parents=True, exist_ok=True)

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
    """Return set of basenames already processed in JSONL (safe even if file doesn't exist yet)."""
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

def append_jsonl(jsonl_path: Path, row: dict):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def print_result_to_console(p: Path, text: str):
    """Prints a short, parse-friendly line for each image."""
    header = f"[{p.name}] "
    if not text:
        print(header + "EMPTY_RESPONSE")
        return
    t = text.strip()
    if t.lower() == "no face":
        print(header + "no face")
    else:
        # Try to keep a single-line JSON if possible
        try:
            obj = json.loads(t)
            print(header + json.dumps(obj, ensure_ascii=False))
        except Exception:
            # fall back to raw (truncated) text
            preview = t.replace("\n", " ")[:500]
            print(header + f"NON_JSON: {preview}")

# ============== OLLAMA CALL ==============
def ollama_generate(prompt: str, image_b64: Optional[str], timeout_sec: int) -> Dict[str, str]:
    """
    Calls Ollama /api/generate with stream=False.
    If image_b64 is provided, passes it via 'images': [<b64>].
    Returns {'text': <model_text>} or raises on HTTP error.
    """
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": MODEL_TAG,
        "prompt": prompt,
        "options": {"temperature": TEMPERATURE},
        "stream": False,
    }
    if image_b64:
        payload["images"] = [image_b64]
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    return {"text": data.get("response", "")}

# ====================== MAIN ======================
def main():
    ensure_dirs()

    # 1) Gather all images and resume state
    images = list_all_images(IMAGES_DIR)
    if not images:
        raise SystemExit(f"No images found in {IMAGES_DIR}")
    done = load_done_set(RESULTS_JSONL)
    todo = [p for p in images if base_filename(str(p)) not in done]

    print(f"Total images: {len(images)} | Already done: {len(done)} | To do: {len(todo)}")
    if len(todo) == 0:
        print("Nothing to do. Proceeding to comparison step...")

    # 2) Process all pending images
    processed = 0
    for idx, p in enumerate(todo, 1):
        b64, _mime = reencode_to_jpeg_b64(p)
        attempt = 0
        row = None
        while attempt < MAX_RETRIES:
            try:
                t0 = time.time()
                resp = ollama_generate(
                    prompt=PROMPT,
                    image_b64=b64,
                    timeout_sec=TIMEOUT_SEC
                )
                latency_ms = int((time.time() - t0) * 1000)
                text = (resp.get("text") or "").strip()

                # ---- Print live to console per image ----
                print_result_to_console(p, text)

                # Accept either 'no face' OR strict JSON. Anything else -> non_json_response
                if text.lower() == "no face":
                    row = {
                        "image_path": str(p),
                        "model": MODEL_TAG,
                        "latency_ms": latency_ms,
                        "error": "no_face",
                        "_raw": text,
                    }
                else:
                    data = safe_json(text)
                    if data is None:
                        row = {
                            "image_path": str(p),
                            "model": MODEL_TAG,
                            "latency_ms": latency_ms,
                            "error": "non_json_response",
                            "_raw": text,
                        }
                    else:
                        # Ensure exactly the six attributes exist; if missing, set None
                        for attr in EXPECTED_ATTRS:
                            data.setdefault(attr, None)
                        row = {"image_path": str(p), "model": MODEL_TAG, "latency_ms": latency_ms, **data}
                break  # success
            except Exception as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    row = {"image_path": str(p), "model": MODEL_TAG, "error": f"exception:{e}"}
                    print(f"[{p.name}] ERROR after {attempt} attempts: {e}")
                else:
                    time.sleep(0.5 * attempt)  # backoff and retry

        append_jsonl(RESULTS_JSONL, row)
        processed += 1
        time.sleep(SLEEP_BETWEEN)

    print(f"\nStreaming results saved to → {RESULTS_JSONL}")

    # 3) COMPARISON — Load results into DataFrame (include any previous runs)
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
        print("No rows parsed from results JSONL. Exiting.")
        return

    df = pd.DataFrame(rows)

    # Split errors safely
    if "error" in df.columns:
        mask_err = df["error"].astype(str).str.len().gt(0) & df["error"].notna()
    else:
        mask_err = pd.Series(False, index=df.index)

    errs = df.loc[mask_err].copy()
    df_good = df.loc[~mask_err].copy()

    if "image_path" not in df_good.columns or df_good.empty:
        print("\nNo valid predictions to score (all lines errored or missing image_path).")
        return

    df_good["_image_file"] = df_good["image_path"].map(base_filename)

    # 4) Load metadata & align columns
    meta = pd.read_csv(METADATA_CSV)
    img_col = detect_image_column(meta)
    if not img_col:
        for c in ["image", "filename", "file", "img", "path", "Image", "Filename"]:
            if c in meta.columns:
                img_col = c
                break
    if not img_col:
        raise SystemExit("Could not detect the image filename column in metadata.csv. Please rename it to 'image'.")

    meta["_image_file"] = meta[img_col].apply(base_filename)

    # Map expected attributes by normalized column names
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

    # 5) Merge and compute accuracy
    merged = pd.merge(df_good, meta_small, on="_image_file", how="inner", suffixes=("_pred", "_gt"))

    summary_rows = []
    mismatches = []
    for attr in EXPECTED_ATTRS:
        pred_col = f"{attr}_pred"
        gt_col   = f"{attr}_gt"
        if pred_col not in merged.columns or gt_col not in merged.columns:
            continue

        sub = merged[["_image_file", pred_col, gt_col]].copy()
        sub["pred_norm"] = sub[pred_col].map(normalize_value)
        sub["gt_norm"]   = sub[gt_col].map(normalize_value)
        sub["match"]     = (sub["pred_norm"] == sub["gt_norm"]) & sub["gt_norm"].ne("")

        total   = int(sub["gt_norm"].ne("").sum())
        correct = int(sub["match"].sum())
        acc     = (correct / total) if total else 0.0

        summary_rows.append({"attribute": attr, "n_scored": total, "n_correct": correct, "accuracy": round(acc, 4)})

        mm = sub[~sub["match"]].copy()
        mm = mm.rename(columns={pred_col: "pred_raw", gt_col: "gt_raw"})
        mm = mm[["_image_file", "pred_raw", "gt_raw", "pred_norm", "gt_norm"]]
        mm.insert(0, "attribute", attr)
        mismatches.append(mm)

    summary = pd.DataFrame(summary_rows).sort_values("attribute")
    mismatch_df = pd.concat(mismatches, ignore_index=True) if mismatches else pd.DataFrame()

    # 6) Save reports
    COMPARE_OUT.mkdir(parents=True, exist_ok=True)
    summary_path        = COMPARE_OUT / "summary.csv"
    mismatches_path     = COMPARE_OUT / "mismatches_head.csv"
    merged_sample_path  = COMPARE_OUT / "merged_sample.csv"
    errors_path         = COMPARE_OUT / "skipped_errors.jsonl"

    summary.to_csv(summary_path, index=False)
    mismatch_df.head(200).to_csv(mismatches_path, index=False)
    merged.head(100).to_csv(merged_sample_path, index=False)

    if not errs.empty:
        with open(errors_path, "w", encoding="utf-8") as f:
            for _, r in errs.iterrows():
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # 7) Print concise summary
    print("\n=== Comparison summary (ALL images) ===")
    if summary.empty:
        print("No comparable attributes found. Ensure your prompt keys match metadata column names exactly.")
    else:
        print(summary.to_string(index=False))

    if not mismatch_df.empty:
        print("\nTop mismatches (head):")
        print(mismatch_df.head(10).to_string(index=False))

    if not errs.empty:
        print(f"\nNote: Skipped {len(errs)} result lines with errors (written to {errors_path}).")

    print(f"\nSaved reports:\n - {summary_path}\n - {mismatches_path}\n - {merged_sample_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
