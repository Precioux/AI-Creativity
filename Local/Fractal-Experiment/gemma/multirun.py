import os
import io
import json
import time
import base64
import requests
from pathlib import Path
from PIL import Image

# ================= CONFIGURATION =================
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
MODEL_NAME = "gemma3"
TEMPERATURES = [0.0, 1.0]
NUM_RUNS = 5
TIMEOUT_SEC = 120

# ================= PATHS =================
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
IMAGE_DIRS = [DATA_ROOT / "FD12", DATA_ROOT / "FD14", DATA_ROOT / "FD16"]

# --- این همون خط جادوییه که اسم فایل خودت توش قرار گرفت ---
OUTPUT_FILE = ROOT / "gemma_fractal_results_10runs_total.jsonl"
# -----------------------------------------------------------

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

Do not explain your answer. Respond with either no face or the JSON only."""


def img_to_b64(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((512, 512))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_ollama(prompt, image_b64, temp):
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": temp,
            "num_predict": 150,
            "num_ctx": 2048
        }
    }
    r = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def main():
    images = []
    for d in IMAGE_DIRS:
        if d.exists():
            images.extend(sorted(list(d.glob("*.png"))) + sorted(list(d.glob("*.jpg"))))

    if not images:
        print(f"ERROR: No images found in {IMAGE_DIRS}")
        return

    done_set = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_set.add((d["file"], d["temperature"], d["run_id"]))
                except:
                    continue

    print("--- Starting Gemma Run (Fractals) ---")
    total_tasks = len(images) * len(TEMPERATURES) * NUM_RUNS
    print(f"Total target: {total_tasks} | Already finished: {len(done_set)}")

    for temp in TEMPERATURES:
        for run_id in range(1, NUM_RUNS + 1):
            for img_p in images:
                if (img_p.name, temp, run_id) in done_set:
                    continue

                try:
                    b64 = img_to_b64(img_p)
                    t0 = time.time()
                    raw_resp = call_ollama(PROMPT, b64, temp)
                    latency = int((time.time() - t0) * 1000)

                    clean_resp = raw_resp.replace("```json", "").replace("```", "").strip()

                    result = {
                        "run_id": run_id,
                        "temperature": temp,
                        "file": img_p.name,
                        "response": clean_resp,
                        "ms": latency
                    }

                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                    print(f"[Temp: {temp} | Run: {run_id}] {img_p.name} -> SUCCESS ({latency}ms)")
                except Exception as e:
                    print(f"WARNING: Error on {img_p.name}: {e}")
                    time.sleep(2)

    print("--- GEMMA RUN COMPLETED ---")


if __name__ == "__main__":
    main()