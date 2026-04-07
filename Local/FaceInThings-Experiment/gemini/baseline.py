
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_and_compare.py

End-to-end script to process ALL images in FacesInThings/images with Gemini,
save streaming JSONL outputs (with resume), and compare against metadata.csv.

What it does:
1) Reads Gemini API key from key.txt (PATHS section below).
2) Iterates over ALL images under FacesInThings/images (with resume).
3) For each image, requests EXACT six attributes as JSON (dataset-aligned).
4) Appends one line per image to out/gemini_all.jsonl as soon as it's ready.
5) After completion, loads metadata.csv and computes per-attribute accuracy.
6) Writes comparison reports to compare_out/ and prints a summary.

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

import pandas as pd
from PIL import Image

# ====================== PATHS — EDIT IF NEEDED ======================
GEMINI_FOLDER   = Path("/Users/precioux/Desktop/Creativity/code/gemini")
KEY_PATH        = GEMINI_FOLDER / "key.txt"

IMAGES_DIR      = Path("/Users/precioux/Desktop/Creativity/FacesInThings/images")
METADATA_CSV    = Path("/Users/precioux/Desktop/Creativity/FacesInThings/metadata.csv")

OUT_BASE        = GEMINI_FOLDER
RESULTS_JSONL   = OUT_BASE / "out/gemini_all.jsonl"
COMPARE_OUT     = OUT_BASE / "compare_out"

# ====================== MODEL CONFIG ======================
MODEL_NAME   = "gemini-2.5-flash-lite"  # swap to "gemini-1.5-flash" if needed
TEMPERATURE  = 0.0
TIMEOUT_SEC  = 60
SLEEP_BETWEEN = 0.10  # polite spacing between calls
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


# ====================== MAIN ======================
def main():
    ensure_dirs()

    # 1) Configure Gemini (REST transport avoids gRPC 504s)
    api_key = read_key(KEY_PATH)
    import google.generativeai as genai
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel(MODEL_NAME)

    # 2) Gather all images and resume state
    images = list_all_images(IMAGES_DIR)
    if not images:
        raise SystemExit(f"No images found in {IMAGES_DIR}")
    done = load_done_set(RESULTS_JSONL)
    todo = [p for p in images if base_filename(str(p)) not in done]

    print(f"Total images: {len(images)} | Already done: {len(done)} | To do: {len(todo)}")
    if len(todo) == 0:
        print("Nothing to do. Proceeding to comparison step...")

    # 3) Process all pending images (sequential, with retry & streaming append)
    processed = 0
    for idx, p in enumerate(todo, 1):
        b64, mime = reencode_to_jpeg_b64(p)
        attempt = 0
        row = None
        while attempt < MAX_RETRIES:
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
                data = safe_json(resp.text)

                if data is None:
                    row = {
                        "image_path": str(p),
                        "model": MODEL_NAME,
                        "latency_ms": latency_ms,
                        "error": "non_json_response",
                        "_raw": resp.text,
                    }
                else:
                    # Ensure exactly the six attributes exist; if missing, set None
                    for attr in EXPECTED_ATTRS:
                        data.setdefault(attr, None)
                    row = {"image_path": str(p), "model": MODEL_NAME, "latency_ms": latency_ms, **data}
                break  # success or non-JSON but got response
            except Exception as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    row = {"image_path": str(p), "model": MODEL_NAME, "error": f"exception:{e}"}
                else:
                    time.sleep(0.5 * attempt)  # backoff and retry

        append_jsonl(RESULTS_JSONL, row)
        processed += 1

        if processed % 25 == 0 or processed == len(todo):
            print(f"Processed {processed}/{len(todo)} | Last: {p.name}")
        time.sleep(SLEEP_BETWEEN)

    print(f"\nStreaming results saved to → {RESULTS_JSONL}")

    # 4) COMPARISON — Load results into DataFrame
    #    (read from JSONL to include any previous runs)
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

    # 5) Load metadata & align columns
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

    # 6) Merge and compute accuracy
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

    # 7) Save reports
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

    # 8) Print concise summary
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
