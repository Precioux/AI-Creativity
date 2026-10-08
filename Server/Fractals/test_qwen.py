import os
import json
import re
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.5-9b"
MAX_TOKENS = 8000

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

def safe_parse_json(raw: str):
    """
    Robust JSON parser: Strips out reasoning tags and extracts raw JSON.
    """
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    
    match = re.search(r'(\{.*\})', cleaned, flags=re.DOTALL)
    
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    try:
        cleaned_fallback = raw.strip()
        if cleaned_fallback.startswith("```"):
            lines = cleaned_fallback.splitlines()
            end = -1 if lines[-1].strip() == "```" else len(lines)
            cleaned_fallback = "\n".join(lines[1:end])
            if cleaned_fallback.lower().startswith("json"):
                cleaned_fallback = cleaned_fallback[4:].strip()
        return json.loads(cleaned_fallback)
    except Exception:
        return None

def run_test():
    print(f"Testing Model: {MODEL}...")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": DAT_PROMPT}],
        "temperature": 1.0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"} 
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        
        raw_content = response.json()["choices"][0]["message"]["content"]
        print("\n" + "="*50)
        print("RAW OUTPUT FROM MODEL (Might contain <think> tags):")
        print("="*50)
        print(raw_content)
        print("="*50 + "\n")
        
        parsed_data = safe_parse_json(raw_content)
        
        print("PARSED JSON RESULT:")
        print("-" * 20)
        if parsed_data:
            print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
            print("\n[SUCCESS] Parser successfully extracted the JSON.")
        else:
            print("[FAILURE] Parser could not extract a valid JSON.")
            
    except Exception as e:
        print(f"[ERROR] API Call Failed: {e}")

if __name__ == "__main__":
    run_test()