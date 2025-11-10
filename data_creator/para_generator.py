import google.generativeai as genai
import time
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY_1")

if not API_KEY:
    raise ValueError("API key not found! Ensure GEMINI_API_KEY is set in .env file")


genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

prompt = """
Generate a long random paragraph (40 words) for typing practice.
It should look natural, NOT meaningful. Mix:
- simple words
- all lowercased
- No punctuations

Do NOT format as bullet points. Output one continuous paragraph.
"""


def generate_text():
    print("[INFO] Requesting Gemini to generate paragraph...")
    start = time.time()
    response = model.generate_content(prompt)
    text = response.text.strip()
    end = time.time()

    print(f"[INFO] Response received in {round(end-start, 2)}s")
    return text

if __name__ == "__main__":
    generate_text()
