#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma-3 Pareidolia Runner — Logging Enabled Version

Processes all PNG images in FD12/FD14/FD16 using Gemma-3 (via Ollama),
streams results to JSONL, and logs detailed progress to console and file.
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

import requests
import pandas as pd
from PIL import Image

# ====================== CONFIG ======================
ROOT = Path(__file__).resolve().parent
GEMMA_FOLDER  = ROOT
IMAGE_DIRS    = [ROOT / "FD12", ROOT / "FD14", ROOT / "FD16"]
METADATA_CSV  = ROOT / "metadata.csv"

OUT_BASE      = GEMMA_FOLDER
RESULTS_JSONL = OUT_BASE / "out/gemma_all.jsonl"
COMPARE_OUT   = OUT_BASE / "compare_out"
LOG_FILE      = OUT_BASE / "out/run.log"

OLLAMA_BASE   = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
MODEL_TAG     = os.environ.get("GEMMA_TAG", "gemma3")
TEMPERATURE   = 0.0
TIMEOUT_SEC   = 60
SLEEP_BETWEEN = 0.10
MAX_RETRIES   = 3

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

# ====================== LOGGING ======================
def setup_logger():
    (OUT_BASE / "out").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Starting run | OLLAMA_BASE={OLLAMA_BASE} | MODEL_TAG={MODEL_TAG}")

# ====================== HELPERS ======================
def list_png_images(dirs: List[Path]) -> List[Path]:
    imgs = []
    for d in dirs:
        if d.exists():
            imgs.extend(sorted([p for p in d.glob("*.png") if p.is_file()]))
    return imgs

def reencode_to_jpeg_b64(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def safe_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text.strip())
    except Exception:
        return None

def load_done_set(jsonl_path: Path) -> set:
    done = set()
    if jsonl_path.exists():
        for line in open(jsonl_path, "r", encoding="utf-8"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                name = Path(obj.get("image_path", "")).name
                if name:
                    done.add(name)
            except json.JSONDecodeError:
                continue
    return done

def append_jsonl(jsonl_path: Path, row: dict):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ====================== OLLAMA CALLS ======================
def ollama_generate_with_image(prompt: str, image_b64: str, timeout_sec: int) -> Dict[str, str]:
    url_gen = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": MODEL_TAG,
        "prompt": prompt,
        "images": [image_b64],
        "options": {"temperature": TEMPERATURE},
        "stream": False,
    }
    r = requests.post(url_gen, json=payload, timeout=timeout_sec)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    data = r.json()
    if "error" in data and data["error"]:
        raise RuntimeError(data["error"])
    return {"text": data.get("response", "")}

# ====================== MAIN ======================
def main():
    setup_logger()

    images = list_png_images(IMAGE_DIRS)
    if not images:
        logging.error(f"No images found in {IMAGE_DIRS}")
        return

    done = load_done_set(RESULTS_JSONL)
    todo = [p for p in images if p.name not in done]

    logging.info(f"Discovered {len(images)} PNG images | Done={len(done)} | Pending={len(todo)}")
    processed = 0

    for idx, p in enumerate(todo, 1):
        b64 = reencode_to_jpeg_b64(p)
        attempt = 0
        row = None

        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                t0 = time.time()
                resp = ollama_generate_with_image(PROMPT, b64, TIMEOUT_SEC)
                latency_ms = int((time.time() - t0) * 1000)
                text = resp.get("text", "").strip()

                if text.lower() == "no face":
                    row = {"image_path": str(p), "model": MODEL_TAG, "latency_ms": latency_ms,
                           "error": "no_face", "_raw": text}
                else:
                    data = safe_json(text)
                    if data is None:
                        row = {"image_path": str(p), "model": MODEL_TAG, "latency_ms": latency_ms,
                               "error": "non_json_response", "_raw": text}
                    else:
                        for attr in EXPECTED_ATTRS:
                            data.setdefault(attr, None)
                        row = {"image_path": str(p), "model": MODEL_TAG, "latency_ms": latency_ms, **data}

                logging.info(f"[{idx}/{len(todo)}] OK {p.name} | {latency_ms}ms")
                break

            except Exception as e:
                logging.warning(f"[{idx}/{len(todo)}] Attempt {attempt} failed for {p.name}: {e}")
                if attempt >= MAX_RETRIES:
                    row = {"image_path": str(p), "model": MODEL_TAG, "error": f"exception:{e}"}
                    break
                time.sleep(0.5 * attempt)

        append_jsonl(RESULTS_JSONL, row)
        processed += 1
        time.sleep(SLEEP_BETWEEN)

    logging.info(f"Completed {processed} images. Results saved to {RESULTS_JSONL}")
    logging.info("Run finished successfully.")

if __name__ == "__main__":
    main()
