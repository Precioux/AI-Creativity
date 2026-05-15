import os
import json
import requests
import base64
from pathlib import Path

# ================= CONFIGURATION =================
MODEL = "gemma3"
CURRENT_FILE = Path(__file__).resolve()
CREATIVITY_ROOT = CURRENT_FILE.parents[4]

IMAGE_DIR = CREATIVITY_ROOT / "FacesInThings" / "sample"
OUTPUT_FILE = CURRENT_FILE.parent / "gemma_faces_results_10runs.jsonl"

PROMPT = """You are given an image. Determine if there is a face.
If no face is visible, respond with: no face
If a face is visible, respond ONLY with this JSON:
{
  "Hard to spot?": "<Easy|Medium|Hard>",
  "Accident or design?": "<Accident|Design>",
  "Emotion?": "<Happy|Neutral|Disgusted|Angry|Surprised|Scared|Sad|Other>",
  "Person or creature?": "<Human-Adult|Human-Old|Human-Young|Cartoon|Animal|Robot|Alien|Other>",
  "Gender?": "<Male|Female|Neutral>",
  "Amusing?": "<Yes|Somewhat|No>"
}"""

# تعریف تنظیمات ۱۰ اجرا (۵ تا دمای ۰، ۵ تا دمای ۱)
RUNS_CONFIG = []
for i in range(1, 6): RUNS_CONFIG.append({"run_id": i, "temp": 0.0})
for i in range(6, 11): RUNS_CONFIG.append({"run_id": i, "temp": 1.0})


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def run_inference(image_path, temp):
    encoded = encode_image(image_path)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "images": [encoded],
            "stream": False,
            "options": {"temperature": temp}
        }
    )
    return response.json().get("response", "").strip()


def main():
    all_images = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
    total_tasks = len(all_images) * 10  # هر عکس ۱۰ بار

    # قابلیت Resume: خواندن کارهایی که قبلاً انجام شده (بر اساس اسم فایل و شماره اجرا)
    processed_tasks = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_tasks.add((data["file"], data["run_id"]))
                except:
                    continue

    print(f"--- Total Tasks: {total_tasks} | Already Done: {len(processed_tasks)} ---")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        current_count = len(processed_tasks)
        for img_p in all_images:
            for config in RUNS_CONFIG:
                run_id = config["run_id"]
                temp = config["temp"]

                # اگر این ترکیب (فایل، شماره اجرا) قبلاً انجام شده، بپر
                if (img_p.name, run_id) in processed_tasks:
                    continue

                try:
                    current_count += 1
                    print(f"[{current_count}/{total_tasks}] Image: {img_p.name} | Run: {run_id} | Temp: {temp}...",
                          end=" ", flush=True)

                    res = run_inference(img_p, temp)

                    result = {
                        "run_id": run_id,
                        "temperature": temp,
                        "file": img_p.name,
                        "response": res
                    }
                    f.write(json.dumps(result) + "\n")
                    f.flush()

                    print("✅")
                except Exception as e:
                    print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()