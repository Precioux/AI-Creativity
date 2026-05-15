import os
import json
import time
import base64
import requests
from pathlib import Path

# ================= CONFIGURATION =================
MODEL_NAME = "gemini-2.5-flash-lite"
TEMPERATURES = [0.0, 1.0]
NUM_RUNS = 5
RPM_LIMIT = 500  # Because you are now PAID!

# ================= DYNAMIC PATHS =================
CURRENT_FILE = Path(__file__).resolve()
CREATIVITY_ROOT = CURRENT_FILE.parents[4]

# Path to your 300 samples: Creativity/FacesInThings/sample
IMAGE_DIR = CREATIVITY_ROOT / "FacesInThings" / "sample"
OUTPUT_FILE = CURRENT_FILE.parent / "gemini_faces_results_10runs.jsonl"

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
}"""


def get_image_data(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_gemini_api(api_key, image_b64, temp):
    url = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_NAME}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
            ]
        }],
        "generationConfig": {"temperature": temp}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    return response.json()['candidates'][0]['content']['parts'][0]['text']


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        return

    images = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))

    done_set = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_set.add((d["file"], d["temperature"], d["run_id"]))
                except:
                    continue

    print(f"--- Starting FacesInThings 300 (Paid Mode) ---")
    for temp in TEMPERATURES:
        for run_id in range(1, NUM_RUNS + 1):
            for img_p in images:
                if (img_p.name, temp, run_id) in done_set: continue

                try:
                    img_b64 = get_image_data(img_p)
                    t0 = time.time()
                    text = call_gemini_api(api_key, img_b64, temp)
                    latency = int((time.time() - t0) * 1000)

                    res = {"run_id": run_id, "temperature": temp, "file": img_p.name, "response": text.strip(),
                           "ms": latency}
                    with open(OUTPUT_FILE, "a") as f:
                        f.write(json.dumps(res) + "\n")

                    print(f"[Temp {temp} | Run {run_id}] {img_p.name} -> SUCCESS")
                    time.sleep(60 / RPM_LIMIT)
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(2)


if __name__ == "__main__":
    main()