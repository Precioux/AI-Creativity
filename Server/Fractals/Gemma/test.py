"""
Quick Pipeline Test
===================
Picks the first available image and runs exactly 9 API calls:

  Phase 1 — DAT Baseline (3 calls)
      temp=0.0  → DAT only, fresh thread
      temp=1.0  → DAT only, fresh thread
      temp=1.5  → DAT only, fresh thread

  Phase 2 — Main experiment (6 calls, 2 per temperature)
      temp=0.0  → fractal prompt + image  |  DAT in same thread
      temp=1.0  → fractal prompt + image  |  DAT in same thread
      temp=1.5  → fractal prompt + image  |  DAT in same thread

Output: test_results.jsonl  (one record per sub-task)

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python test_pipeline.py
"""

import os
import io
import sys
import json
import time
import base64
from pathlib import Path

import requests
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL    = "google/gemma-4-26b-a4b-it"

TEMPERATURES    = [0.0]
TEMP_LABELS     = {0.0: "deterministic", 1.0: "default", 1.5: "high-creative"}
RETRY_BACKOFF   = [20, 45, 90]
INTER_CALL_SLEEP = 30  # seconds between calls (paid tier handles 60+ RPM)

ROOT      = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "Fractals" / "new-FD-images"
IMAGE_DIRS = [DATA_ROOT / "FD135", DATA_ROOT / "FD17"]   # ← update if directory names differ
OUTPUT    = ROOT / "test_results.jsonl"

# =============================================================================
# PROMPTS
# =============================================================================
FRACTAL_PROMPT = (
    "Look at the image. Try to find hidden shapes or objects. "
    "Describe each percept you found using one word per percept. "
    "If the single word doesn't capture well what you've seen, you can add an optional description. "
    "You will have to tell me where in the image you found the percept later. "
    "Output must be strictly valid JSON only, with no extra text.\n"
    "Schema:\n"
    '{"percepts":[{"label":"string","description":"string","confidence_present":0}]}\n'
    "Constraints:\n"
    '- "label" must be exactly one word.\n'
    '- "description" must be a string. If unnecessary, use "".\n'
    '- "confidence_present" must be a number from 0 to 100 indicating how confident you are '
    "that the percept is actually present in the image.\n"
    '- If nothing is found, return {"percepts":[]}.\n'
    "- Do not mention coordinates, regions, or explanations outside the JSON.\n"
    "- Do not output markdown."
)

DAT_PROMPT = (
    "Please enter 10 words that are as different from each other as possible, "
    "in all meanings and uses of the words. "
    "Rules: Only single words in English. "
    "Only nouns (e.g., things, objects, concepts). "
    "No proper nouns (e.g., no specific people or places). "
    "No specialised vocabulary (e.g., no technical terms). "
    "Think of the words on your own (e.g., do not just look at objects in your surroundings). "
    "Make a list of these 10 words, a single word in each entry of the list. "
    "Output must be strictly valid JSON only, with no extra text.\n"
    'Schema:\n{"words":["string"]}'
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


def image_block(b64: str, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def call_openrouter(messages: list[dict], temperature: float) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fractal-pareidolia-pipeline",
        "X-Title": "Fractal Pareidolia Test",
    }
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}

    last_exc = None
    for attempt, backoff in enumerate(RETRY_BACKOFF, start=1):
        try:
            r = requests.post(BASE_URL, headers=headers, json=payload, timeout=90)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            print(f"    Attempt {attempt}/{len(RETRY_BACKOFF)} failed: {exc}")
            if attempt < len(RETRY_BACKOFF):
                print(f"    Waiting {backoff}s before retry...")
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


def save(record: dict) -> None:
    with open(OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def pick_image() -> Path:
    for d in IMAGE_DIRS:
        if d.exists():
            candidates = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
            if candidates:
                return candidates[0]
    sys.exit(f"ERROR: no images found in {IMAGE_DIRS}")


def divider(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not OPENROUTER_API_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY is not set.")

    image = pick_image()
    b64, mime = img_to_b64(image)

    print(f"\n{'='*55}")
    print(f"  PIPELINE QUICK TEST")
    print(f"  Model : {MODEL}")
    print(f"  Image : {image.name}  ({image.parent.name})")
    print(f"  Temps : {TEMPERATURES}")
    print(f"  Total : 9 API calls  (3 baseline + 6 main)")
    print(f"  Sleep : {INTER_CALL_SLEEP}s between calls")
    print(f"  Out   : {OUTPUT.name}")
    print(f"{'='*55}")

    # ─────────────────────────────────────────────────────────
    # PHASE 1 — DAT Baseline  (3 calls, one per temperature)
    # ─────────────────────────────────────────────────────────
    divider("PHASE 1 — DAT Baseline  (no image, fresh thread)")

    for temp in TEMPERATURES:
        label = TEMP_LABELS[temp]
        print(f"\n  → temp={temp}  [{label}]")

        messages = [{"role": "user", "content": DAT_PROMPT}]
        try:
            t0  = time.time()
            raw = call_openrouter(messages, temperature=temp)
            ms  = int((time.time() - t0) * 1000)

            parsed = safe_parse_json(raw)
            words  = parsed.get("words", []) if parsed else None

            record = {
                "phase"      : "baseline",
                "temperature": temp,
                "temp_label" : label,
                "dat_raw"    : raw,
                "dat_parsed" : parsed,
                "ms"         : ms,
            }
            save(record)

            status = "✓ parsed" if parsed else "✗ parse error"
            print(f"    {status}  |  {ms} ms")
            print(f"    words: {words}")

        except Exception as exc:
            print(f"    FAILED: {exc}")

        time.sleep(INTER_CALL_SLEEP)

    # ─────────────────────────────────────────────────────────
    # PHASE 2 — Main  (6 calls: 2 per temperature)
    # ─────────────────────────────────────────────────────────
    divider(f"PHASE 2 — Main  (fractal → DAT, same thread)  [{image.name}]")

    for temp in TEMPERATURES:
        label = TEMP_LABELS[temp]
        print(f"\n  → temp={temp}  [{label}]")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FRACTAL_PROMPT},
                    image_block(b64, mime),
                ],
            }
        ]

        try:
            t0          = time.time()
            fractal_raw = call_openrouter(messages, temperature=temp)
            ms_fractal  = int((time.time() - t0) * 1000)
            time.sleep(INTER_CALL_SLEEP)

            fractal_parsed = safe_parse_json(fractal_raw)
            n = len(fractal_parsed["percepts"]) if fractal_parsed and "percepts" in fractal_parsed else "?"
            f_status = "✓ parsed" if fractal_parsed else "✗ parse error"
            print(f"    Fractal  {f_status}  |  {n} percepts  |  {ms_fractal} ms")

            # Turn 2: DAT in same thread
            messages.append({"role": "assistant", "content": fractal_raw})
            messages.append({"role": "user",      "content": DAT_PROMPT})

            t0      = time.time()
            dat_raw = call_openrouter(messages, temperature=temp)
            ms_dat  = int((time.time() - t0) * 1000)

            dat_parsed = safe_parse_json(dat_raw)
            words      = dat_parsed.get("words", []) if dat_parsed else None
            d_status   = "✓ parsed" if dat_parsed else "✗ parse error"
            print(f"    DAT      {d_status}  |  {ms_dat} ms")
            print(f"    words: {words}")

            record = {
                "phase"         : "main",
                "temperature"   : temp,
                "temp_label"    : label,
                "file"          : image.name,
                "fd"            : image.parent.name,
                "fractal_raw"   : fractal_raw,
                "fractal_parsed": fractal_parsed,
                "ms_fractal"    : ms_fractal,
                "dat_raw"       : dat_raw,
                "dat_parsed"    : dat_parsed,
                "ms_dat"        : ms_dat,
            }
            save(record)

        except Exception as exc:
            print(f"    FAILED: {exc}")

        time.sleep(INTER_CALL_SLEEP)

    print(f"\n{'='*55}")
    print(f"  TEST COMPLETE — results in {OUTPUT.name}")
    print(f"  Check: JSON parse status, temp=1.5 acceptance,")
    print(f"         response quality, and latencies.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()