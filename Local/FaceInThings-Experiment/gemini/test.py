# test_gemini_sdk.py
import google.generativeai as genai

api_key = open("key.txt").read().strip()

# 👇 Force REST transport (avoids gRPC timeouts on some networks)
genai.configure(api_key=api_key, transport="rest")

MODEL = "gemini-1.5-flash"   # safe, widely available
model = genai.GenerativeModel(MODEL)

resp = model.generate_content(
    "Say 'pong' if you can read this.",
    request_options={"timeout": 30, "retry": None},  # fail fast
)
print("Model:", MODEL)
print("Reply:", resp.text)
