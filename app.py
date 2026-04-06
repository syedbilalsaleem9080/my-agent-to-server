import os
from flask import Flask
import google.generativeai as genai
from groq import Groq
from openai import OpenAI

app = Flask(__name__)

# --- API Keys Setup (Render settings se ayengi) ---
# Gemini Setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Groq Setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# DeepSeek Setup (OpenAI library use karta hai)
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

@app.route('/')
def home():
    return """
    <h1>AI Multi-Agent Server is Live!</h1>
    <p>Test the following routes:</p>
    <ul>
        <li><b>/gemini</b> - Google's AI</li>
        <li><b>/groq</b> - Fast Llama 3</li>
        <li><b>/deepseek</b> - DeepSeek V3</li>
    </ul>
    """

@app.route('/gemini')
def call_gemini():
    try:
        model = genai.GenerativeModel('gemini-pro')
        res = model.generate_content("Say 'Gemini is Online!'")
        return f"<h3>Gemini Response:</h3> {res.text}"
    except Exception as e:
        return f"Gemini Error: {str(e)}"

@app.route('/groq')
def call_groq():
    try:
        comp = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Say 'Groq is Online!'"}]
        )
        return f"<h3>Groq Response:</h3> {comp.choices[0].message.content}"
    except Exception as e:
        return f"Groq Error: {str(e)}"

@app.route('/deepseek')
def call_deepseek():
    try:
        comp = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Say 'DeepSeek is Online!'"}]
        )
        return f"<h3>DeepSeek Response:</h3> {comp.choices[0].message.content}"
    except Exception as e:
        return f"DeepSeek Error: {str(e)} <br> (Note: DeepSeek requires paid balance)"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
