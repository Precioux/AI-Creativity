#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_qwen_unlabeled_onefile.py

Processes *all unlabeled* image folders (e.g., FD12, FD14, FD16) under:
  /mnt/upschrimpf2/scratch/mahdipou/test/code/fractal/

Finds all images recursively inside those folders, runs Qwen (via Ollama),
and saves EVERYTHING in a single JSONL file: out/qwen_all.jsonl

You can safely stop & rerun — it will skip already processed images.
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

# ====================== PATHS ======================
BASE_FOLDER = Path("/mnt/mahdipou/test/code/fractal").resolve()
OUT_DIR     = BASE_FOLDER / "out"
OUT_FILE    = OUT_DIR / "qwen_all.jsonl"
OUT_CSV     = OUT_DIR / "qwen_all.csv"

# ====================== MODEL CONFIG ======================
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
MODEL_TAG   = os.environ.get("QWEN_TAG") or "qwen2.5vl"

TEMPERATURE   = 0.0
TIMEOUT_SEC   = 120
SLEEP_BETWEEN = 0.05
MAX_RETRIES   = 3

# ====================== PROMPT ======================
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def list_dataset_folders() -> List[Path]:
    """All subfolders like FD12, FD14, FD16 next to this script."""
    skip = {"out", "__pycache__", "compare_out"}
    ds = []
    for p in BASE_FOLDER.iterdir():
        if p.is_dir() and p.name not in skip and not p.name.startswith("."):
            ds.append(p)
    return sorted(ds)

def list_all_images(root: Path) -> List[Path]:
    """Return all images under root (recursively)."""
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

def reencode_to_jpeg_b64(path: Path) -> Tuple[str, str]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

def safe_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except Exception:
        return None

def base_filename(p: str) -> str:
    return Path(p).name

def load_done_set(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                img = obj.get("image_path") or obj.get("path")
                if img:
                    done.add(base_filename(img))
            except Exception:
                continue
    return done

def append_jsonl(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def print_result_to_console(p: Path, text: str):
    head = f"[{p.name}] "
    if not text:
        print(head + "EMPTY_RESPONSE")
    elif text.strip().lower() == "no face":
        print(head + "no face")
    else:
        try:
            j = json.loads(text)
            print(head + json.dumps(j, ensure_ascii=False))
        except Exception:
            print(head + "NON_JSON")

def ollama_generate(prompt: str, image_b64: str, timeout_sec: int) -> str:
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": MODEL_TAG,
        "prompt": prompt,
        "options": {"temperature": TEMPERATURE},
        "stream": False,
        "images": [image_b64],
    }
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    return r.json().get("response", "")

# ====================== MAIN ======================
def main():
    ensure_dirs()
    datasets = list_dataset_folders()
    if not datasets:
        raise SystemExit("No dataset folders found beside baseline.py")

    # Gather all images from all datasets
    all_images = []
    for ds in datasets:
        imgs = list_all_images(ds)
        if imgs:
            all_images.extend(imgs)
    if not all_images:
        raise SystemExit("No images found in any dataset folders.")

    done = load_done_set(OUT_FILE)
    todo = [p for p in all_images if base_filename(str(p)) not in done]

    print(f"Total datasets: {len(datasets)}")
    print(f"Total images: {len(all_images)} | Already done: {len(done)} | To do: {len(todo)}")

    processed = 0
    for idx, p in enumerate(todo, 1):
        b64, _mime = reencode_to_jpeg_b64(p)
        attempt = 0
        row = None
        while attempt < MAX_RETRIES:
            try:
                t0 = time.time()
                text = ollama_generate(PROMPT, b64, TIMEOUT_SEC)
                latency = int((time.time() - t0) * 1000)
                print_result_to_console(p, text)

                if text.lower().strip() == "no face":
                    row = {"image_path": str(p), "model": MODEL_TAG, "latency_ms": latency, "error": "no_face", "_raw": text}
                else:
                    data = safe_json(text)
                    if data is None:
                        row = {"image_path": str(p), "model": MODEL_TAG, "latency_ms": latency, "error": "non_json", "_raw": text}
                    else:
                        for attr in EXPECTED_ATTRS:
                            data.setdefault(attr, None)
                        row = {"image_path": str(p), "model": MODEL_TAG, "latency_ms": latency, **data}
                break
            except Exception as e:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    row = {"image_path": str(p), "model": MODEL_TAG, "error": f"exception:{e}"}
                    print(f"[{p.name}] ERROR after {attempt} attempts: {e}")
                else:
                    time.sleep(0.5 * attempt)
        append_jsonl(OUT_FILE, row)
        processed += 1
        time.sleep(SLEEP_BETWEEN)

    print(f"\nAll results saved to {OUT_FILE}")

    # Optional: build CSV summary
    try:
        rows = []
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if rows:
            df = pd.DataFrame(rows)
            keep = ["image_path", "model", "latency_ms", "error"] + [c for c in EXPECTED_ATTRS if c in df.columns]
            if "_raw" in df.columns and "_raw" not in keep:
                keep.append("_raw")
            df[keep].to_csv(OUT_CSV, index=False)
            print(f"Summary CSV saved → {OUT_CSV}")
    except Exception as e:
        print(f"CSV export skipped: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
