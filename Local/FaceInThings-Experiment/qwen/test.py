import base64, requests, json

img_path = "/mnt/mahdipou/test/FacesInThings/images/000000009.jpg"
with open(img_path, "rb") as f:
    img64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "qwen2.5vl",
    "prompt": """You are given an image. Look at the image and determine whether there is a face visible.

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

Do not explain your answer. Respond with either no face or the JSON only.""",
    "images": [img64],
    "stream": False
}

resp = requests.post("http://localhost:11434/api/generate", json=payload)
print(resp.json()["response"])

