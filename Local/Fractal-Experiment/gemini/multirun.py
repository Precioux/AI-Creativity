import os
import io
import json
import time
import base64
from pathlib import Path
import google.generativeai as genai
from PIL import Image

# ================= CONFIGURATION =================
# Use the explicit v1 model path to bypass v1beta routing issues
MODEL_NAME = "gemini-2.5-flash-lite"
TEMPERATURES = [0.0, 1.0]
NUM_RUNS = 5
RPM_LIMIT = 500
MAX_RETRIES = 5

# ================= PATHS =================
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
IMAGE_DIRS = [DATA_ROOT / "FD12", DATA_ROOT / "FD14", DATA_ROOT / "FD16"]
OUTPUT_FILE = ROOT / "gemini_fractal_results_10runs_total.jsonl"

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
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        return

    # Initialize with specific API version override
    genai.configure(api_key=api_key)

    # Direct model initialization
    model = genai.GenerativeModel(model_name=MODEL_NAME)

    images = []
    for d in IMAGE_DIRS:
        if d.exists():
            images.extend(sorted(list(d.glob("*.png"))))
            images.extend(sorted(list(d.glob("*.jpg"))))

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

    total_tasks = len(images) * len(TEMPERATURES) * NUM_RUNS
    print(f"--- Starting Gemini Final Run ---")
    print(f"Model: {MODEL_NAME} | Total tasks: {total_tasks} | Already finished: {len(done_set)}")

    for temp in TEMPERATURES:
        for run_id in range(1, NUM_RUNS + 1):
            for img_p in images:
                if (img_p.name, temp, run_id) in done_set:
                    continue

                try:
                    b64, mime = img_to_b64(img_p)
                    t0 = time.time()

                    # Using the direct generate_content method
                    response = model.generate_content(
                        contents=[
                            {"role": "user",
                             "parts": [{"text": PROMPT}, {"inline_data": {"mime_type": mime, "data": b64}}]}
                        ],
                        generation_config={"temperature": temp}
                    )

                    latency = int((time.time() - t0) * 1000)
                    clean_resp = response.text.replace("```json", "").replace("```", "").strip()

                    result = {
                        "run_id": run_id,
                        "temperature": temp,
                        "file": img_p.name,
                        "response": clean_resp,
                        "ms": latency
                    }

                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                    print(f"[Temp: {temp} | Run: {run_id}] {img_p.name} -> SUCCESS")
                    time.sleep(60 / RPM_LIMIT)

                except Exception as e:
                    print(f"WARNING: Error on {img_p.name}: {e}")
                    # If it's a model error, try a 10s backoff
                    time.sleep(10)

    print("--- GEMINI FINAL RUN COMPLETED ---")


if __name__ == "__main__":
    main()