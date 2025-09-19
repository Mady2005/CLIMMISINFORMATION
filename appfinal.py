import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import os
import requests
import json

# --- 1. Load the Fine-Tuned ClimateBERT Model ---
model_path = "climatebert_model"
print("Loading fine-tuned ClimateBERT model...")

if not os.path.isdir(model_path):
    print("--- FATAL ERROR: ClimateBERT model not found ---")
    exit()

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print("✅ ClimateBERT model loaded successfully!")
except Exception as e:
    print(f"--- FATAL ERROR: Could not load ClimateBERT model. Error: {e} ---")
    exit()

# --- 2. Function to Call Google Gemini API ---
def get_generative_explanation(classification, article_text):
    """
    Calls the Google Gemini API to get a simple, educational explanation.
    """
    print(f"Calling Gemini to explain '{classification}'...")

    # --- SECURE KEY HANDLING ---
    # Load the API key from an environment variable for security.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("--- FATAL ERROR: GEMINI_API_KEY not found ---")
        return "Error: GEMINI_API_KEY environment variable not set. Please set the key in your terminal before running."
    # ---

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    prompt = f"""
    Act as a helpful media literacy assistant for a user in India.
    My analysis model has classified the following article as '{classification}'.
    Your task is to:
    1.  Briefly explain what '{classification}' means in one simple sentence.
    2.  Point out one or two general red flags from the article text that might align with this classification.
    3.  Keep the entire explanation under 60 words.
    4.  Write the entire response in simple, clear Hinglish.

    Here is the article text:
    ---
    {article_text}
    ---
    """

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        explanation = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Sorry, I couldn't generate an explanation right now.")
        print("✅ Gemini explanation received.")
        return explanation
    except requests.exceptions.RequestException as e:
        print(f"--- ERROR: API call failed. {e} ---")
        return "Error: Could not connect to the Generative AI service. Check your internet connection and API key."
    except (KeyError, IndexError) as e:
        print(f"--- ERROR: Could not parse API response. {e} ---")
        return "Error: The AI explanation response was in an unexpected format."

# --- 3. Main Prediction Function ---
def full_analysis(title, text):
    """
    Runs the full analysis pipeline: classification + generative explanation.
    """
    if not title and not text:
        return "Awaiting input...", "Awaiting input..."

    combined_text = str(title) + ' ' + re.sub(re.compile('<.*?>'), '', str(text))

    prediction = classifier(combined_text)[0]
    label = prediction['label']
    score = prediction['score']
    classification_output = f"Predicted Source: '{label}' (Confidence: {score:.2%})"

    generative_explanation = get_generative_explanation(label, combined_text[:1000])

    return classification_output, generative_explanation

# --- 4. Gradio Interface ---
print("Launching final Gradio interface...")

iface = gr.Interface(
    fn=full_analysis,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter article title here...", label="Article Title"),
        gr.Textbox(lines=10, placeholder="Paste article text here...", label="Article Text")
    ],
    outputs=[
        gr.Label(label="Classification Result"),
        gr.Textbox(lines=5, label="AI-Generated Explanation (from Google Gemini)")
    ],
    title="Climate Intelligence Assistant (ft. ClimateBERT + Google Gemini)",
    description="This tool analyzes an article to predict its source using a fine-tuned ClimateBERT model, then uses Google's Gemini to provide a simple, educational explanation of the result.",
    examples=[
        [
            "Groundbreaking Study Shows Solar Panels Now Work at Night",
            "<p>In an astonishing leap for renewable energy, scientists have unveiled a new type of photovoltaic cell that generates electricity from moonlight. The study was published in the journal 'Radical Science Tomorrow'.</p>"
        ]
    ]
)

# --- 5. Launch the App ---
if __name__ == "__main__":
    iface.launch()

    

