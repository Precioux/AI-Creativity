"""
Fractal Pareidolia + DAT Priming Pipeline — GPT-5.4
===========================================================
Identical protocol to the Gemma 4 run. Only MODEL and RESULTS_DIR differ.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python fractal_pipeline_gemini31pro.py                   # both phases
    python fractal_pipeline_gemini31pro.py --phase baseline  # Phase 1 only
    python fractal_pipeline_gemini31pro.py --phase main      # Phase 2 only
"""

import os
import io
import sys
import json
import time
import base64
import argparse
from pathlib import Path

import requests
from PIL import Image

# =============================================================================
# CONFIGURATION — only MODEL and RESULTS_DIR differ from Gemma run
# =============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openai/gpt-5.4"

TEMPERATURES      = [0.0, 1.0, 1.5]
NUM_RUNS          = 10
DAT_BASELINE_RUNS = 10

INTER_CALL_SLEEP = 8     # kept identical to Gemma run
RETRY_BACKOFF    = [30, 60, 120]   # kept identical to Gemma run

# =============================================================================
# PATHS
# =============================================================================
ROOT       = Path(__file__).resolve().parent
IMAGE_DIRS = [ROOT / "new-FD-images" /"FD135", ROOT / "new-FD-images" /"FD17"]

RESULTS_DIR     = ROOT / "results" / "GPT-5.4"
BASELINE_OUTPUT = RESULTS_DIR / "dat_baseline_results.jsonl"
MAIN_OUTPUT     = RESULTS_DIR / "main_results.jsonl"

# =============================================================================
# PROMPTS  (identical to Gemma run — do not modify)
# =============================================================================
FRACTAL_PROMPT = ('''
Look at the image.
Try to find hidden shapes or objects.
Describe each percept you found using one word per percept.
If the single word doesn't capture well what you've seen, you can add an optional description.
You will have to tell me where in the image you found the percept later. Output must be strictly valid JSON only, with no extra text.
Schema:
{"percepts":[{"label":"string","description":"string","confidence_present":0}]}
Constraints:
- "label" must be exactly one word.
- "description" must be a string. If unnecessary, use "".
- "confidence_present" must be a number from 0 to 100 indicating how confident you are that the percept is actually present in the image.
- If nothing is found, return {"percepts":[]}.
- Do not mention coordinates, regions, or explanations outside the JSON.
- Do not output markdown.
'''
)

DAT_PROMPT = (
'''
Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings).  Make a list of these 10 words, a single word in each entry of the list.
Output must be strict valid JSON only, with no extra text.
Schema:
{"words":["string","string","string","string","string","string","string","string","string","string"]}
Constraints:
- The array must contain exactly 10 entries.
- Do not output numbering, explanations, or markdown.
'''
)

# =============================================================================
# UTILITIES
# =============================================================================

def img_to_b64(path: Path) -> tuple[str, str]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def image_content_block(b64: str, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def call_openrouter(messages: list[dict], temperature: float) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fractal-pareidolia-pipeline",
        "X-Title": "Fractal Pareidolia Pipeline",
    }
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}

    last_exc = None
    for attempt, backoff in enumerate(RETRY_BACKOFF, start=1):
        try:
            r = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            print(f"    Attempt {attempt}/{len(RETRY_BACKOFF)} failed: {exc}")
            if attempt < len(RETRY_BACKOFF):
                print(f"    Waiting {backoff}s...")
                time.sleep(backoff)

    raise RuntimeError(f"All retries exhausted. Last error: {last_exc}")


def safe_parse_json(raw: str):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[1:end])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def load_done_set(path: Path, key_fields: list[str]) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(tuple(rec[k] for k in key_fields))
            except Exception:
                continue
    return done


def append_result(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_images() -> list[Path]:
    images = []
    for d in IMAGE_DIRS:
        if d.exists():
            images.extend(sorted(d.glob("*.png")))
            images.extend(sorted(d.glob("*.jpg")))
        else:
            print(f"  WARNING: directory not found — {d}")
    return images


def temp_label(t: float) -> str:
    return {0.0: "deterministic", 1.0: "default", 1.5: "high-creative"}.get(t, str(t))


# =============================================================================
# PHASE 1 — DAT BASELINE
# =============================================================================

def run_baseline() -> None:
    KEY_FIELDS = ["temperature", "run_id"]
    done_set   = load_done_set(BASELINE_OUTPUT, KEY_FIELDS)

    total = len(TEMPERATURES) * DAT_BASELINE_RUNS
    print(f"\n{'='*60}")
    print(f"  PHASE 1 — DAT BASELINE")
    print(f"  Model        : {MODEL}")
    print(f"  Temperatures : {TEMPERATURES}")
    print(f"  Runs/temp    : {DAT_BASELINE_RUNS}")
    print(f"  Total calls  : {total}  |  Done: {len(done_set)}")
    print(f"  Output       : {BASELINE_OUTPUT}")
    print(f"{'='*60}\n")

    for temp in TEMPERATURES:
        for run_id in range(1, DAT_BASELINE_RUNS + 1):
            key = (temp, run_id)
            if key in done_set:
                continue

            messages = [{"role": "user", "content": DAT_PROMPT}]
            try:
                t0     = time.time()
                raw    = call_openrouter(messages, temperature=temp)
                ms     = int((time.time() - t0) * 1000)
                parsed = safe_parse_json(raw)
                words  = parsed.get("words", []) if parsed else ["(parse error)"]

                record = {
                    "temperature" : temp,
                    "temp_label"  : temp_label(temp),
                    "run_id"      : run_id,
                    "dat_raw"     : raw,
                    "dat_parsed"  : parsed,
                    "ms"          : ms,
                }
                append_result(BASELINE_OUTPUT, record)
                done_set.add(key)
                print(f"  [Baseline | temp={temp} | run {run_id:02d}]  {words}  ({ms} ms)")

            except Exception as exc:
                print(f"  ERROR [baseline temp={temp} run={run_id}]: {exc}")

            time.sleep(INTER_CALL_SLEEP)


# =============================================================================
# PHASE 2 — MAIN EXPERIMENT
# =============================================================================

def run_main(images: list[Path]) -> None:
    KEY_FIELDS = ["temperature", "file", "run_id"]
    done_set   = load_done_set(MAIN_OUTPUT, KEY_FIELDS)

    total = len(images) * len(TEMPERATURES) * NUM_RUNS
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — MAIN EXPERIMENT  (fractal → DAT, same thread)")
    print(f"  Model        : {MODEL}")
    print(f"  Images       : {len(images)} (FD135 + FD17)")
    print(f"  Temperatures : {TEMPERATURES}")
    print(f"  Runs/image   : {NUM_RUNS}")
    print(f"  Total runs   : {total}  |  Done: {len(done_set)}")
    print(f"  API calls    : {total * 2} total  (2 per run)")
    print(f"  Output       : {MAIN_OUTPUT}")
    print(f"{'='*60}\n")

    for temp in TEMPERATURES:
        for img_p in images:
            for run_id in range(1, NUM_RUNS + 1):
                key = (temp, img_p.name, run_id)
                if key in done_set:
                    continue

                b64, mime = img_to_b64(img_p)

                # Turn 1: pareidolia
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": FRACTAL_PROMPT},
                            image_content_block(b64, mime),
                        ],
                    }
                ]

                try:
                    t0           = time.time()
                    fractal_raw  = call_openrouter(messages, temperature=temp)
                    ms_fractal   = int((time.time() - t0) * 1000)
                    time.sleep(INTER_CALL_SLEEP)

                    fractal_parsed = safe_parse_json(fractal_raw)
                    n_percepts = (
                        len(fractal_parsed["percepts"])
                        if fractal_parsed and "percepts" in fractal_parsed
                        else "?"
                    )

                    # Turn 2: DAT in same thread
                    messages.append({"role": "assistant", "content": fractal_raw})
                    messages.append({"role": "user",      "content": DAT_PROMPT})

                    t0      = time.time()
                    dat_raw = call_openrouter(messages, temperature=temp)
                    ms_dat  = int((time.time() - t0) * 1000)

                    dat_parsed = safe_parse_json(dat_raw)
                    words      = dat_parsed.get("words", []) if dat_parsed else ["(parse error)"]

                    record = {
                        "temperature"    : temp,
                        "temp_label"     : temp_label(temp),
                        "file"           : img_p.name,
                        "fd"             : img_p.parent.name,
                        "run_id"         : run_id,
                        "fractal_raw"    : fractal_raw,
                        "fractal_parsed" : fractal_parsed,
                        "ms_fractal"     : ms_fractal,
                        "dat_raw"        : dat_raw,
                        "dat_parsed"     : dat_parsed,
                        "ms_dat"         : ms_dat,
                    }
                    append_result(MAIN_OUTPUT, record)
                    done_set.add(key)

                    print(
                        f"  [Main | temp={temp} | {img_p.parent.name} | run {run_id:02d}]  "
                        f"{img_p.name}  →  {n_percepts} percepts  |  DAT: {words}  "
                        f"({ms_fractal}+{ms_dat} ms)"
                    )

                except Exception as exc:
                    print(f"  ERROR [temp={temp} {img_p.name} run={run_id}]: {exc}")

                time.sleep(INTER_CALL_SLEEP)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Fractal Pareidolia + DAT Priming Pipeline — GPT-5.4")
    parser.add_argument(
        "--phase",
        choices=["baseline", "main", "both"],
        default="both",
        help="Which phase to run (default: both — baseline always runs first)",
    )
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY environment variable is not set.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    images = collect_images()
    if not images:
        sys.exit(f"ERROR: No images found in {IMAGE_DIRS}")

    print(f"\nFound {len(images)} images across FD135 + FD17")
    print(f"Model  : {MODEL}")
    print(f"Temps  : {TEMPERATURES}  ({[temp_label(t) for t in TEMPERATURES]})")
    print(f"Output : {RESULTS_DIR}")

    if args.phase in ("baseline", "both"):
        run_baseline()

    if args.phase in ("main", "both"):
        run_main(images)

    print("\n=== ALL PHASES COMPLETED ===")


if __name__ == "__main__":
    main()