import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import os
import time

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

# --- 2. SIMULATED Generative AI Function ---
# This function pretends to be the Gemini API. It returns a pre-written explanation.
def get_simulated_explanation(classification):
    """
    Returns a high-quality, pre-written explanation for a given classification.
    This makes the demo reliable when the live API is down.
    """
    print(f"Generating SIMULATED explanation for '{classification}'...")
    
    # A dictionary of pre-written, high-quality responses
    explanations = {
        "news": "Yeh ek news article lagta hai, jo aam taur par haal ki ghatnao par objective reporting karta hai. Ismein facts aur alag alag viewpoints ho sakte hain.",
        "science": "Yeh ek science-based article lagta hai, jo research, data, aur evidence par focus karta hai. Aise articles usually scientific journals ya vishvasniya sansthao se aate hain.",
        "junksci": "Junk science ka matlab hai aisi information jo scientific jaisi dikhti hai, par uske peeche aasal evidence nahi hota. Ismein aksar sensational daave aur 'secret knowledge' ki baatein hoti hain.",
        "conspiracy": "Yeh ek conspiracy theory jaisa lagta hai. Aise articles aksar ek powerful group ko doshi thehrate hain aur mainstream explanations ko ek 'cover-up' batate hain.",
        "opinion": "Yeh ek opinion piece lagta hai, jo writer ke personal views ko express karta hai. Yeh facts par aadharit ho sakta hai, lekin iska mukhya uddeshya personal vishleshan dena hai."
    }
    
    # Add a small delay to simulate a real API call
    time.sleep(2) 
    
    # Return the explanation for the given classification, or a default message
    return explanations.get(classification, "Could not generate an explanation for this category.")


# --- 3. Main Prediction Function ---
def full_analysis(title, text):
    """
    Runs the full analysis pipeline: classification + SIMULATED generative explanation.
    """
    if not title and not text:
        return "Awaiting input...", "Awaiting input..."

    combined_text = str(title) + ' ' + re.sub(re.compile('<.*?>'), '', str(text))
    
    # Step 1: Get prediction from local ClimateBERT model
    prediction = classifier(combined_text)[0]
    label = prediction['label']
    score = prediction['score']
    classification_output = f"Predicted Source: '{label}' (Confidence: {score:.2%})"
    
    # Step 2: Get SIMULATED explanation
    generative_explanation = get_simulated_explanation(label)

    return classification_output, generative_explanation + "\n\n(Note: This is a simulated AI response for demo purposes.)"

# --- 4. Gradio Interface ---
print("Launching final Gradio interface with SIMULATED AI...")

iface = gr.Interface(
    fn=full_analysis,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter article title here...", label="Article Title"),
        gr.Textbox(lines=10, placeholder="Paste article text here...", label="Article Text")
    ],
    outputs=[
        gr.Label(label="Classification Result"),
        gr.Textbox(lines=5, label="AI-Generated Explanation (Simulated)")
    ],
    title="Climate Intelligence Assistant (ft. ClimateBERT + Google Gemini)",
    description="This tool analyzes an article to predict its source using a fine-tuned ClimateBERT model, then uses a simulated AI to provide a simple, educational explanation of the result.",
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

    

